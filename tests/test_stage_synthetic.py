"""End-to-end refinement on synthetic local OME-Zarr tiles with a known placement error."""

from pathlib import Path

import numpy as np
import pytest
import zarr
from scipy.ndimage import gaussian_filter

from block_affine import BlockAffineConfig, qc, run, xml_io

TILE, STRIDE = 128, 80
VOXEL_XYZ = (0.5, 0.5, 1.0)
# Seed placement errors in voxels; the anchor X0Y0 is exact. Perfectly aligned seams
# would be rejected by the "correlation must improve" gate, as in real data.
ERRORS_XYZ = {
    (1, 0): np.array([1.5, 1.0, -0.5]),
    (0, 1): np.array([-1.0, 2.0, 0.5]),
    (1, 1): np.array([2.0, -1.5, 1.0]),
}


def _registration(setup: int, stitch: np.ndarray, nominal_xyz) -> str:
    nominal = np.eye(4)
    nominal[:3, 3] = nominal_xyz
    return (
        f'<ViewRegistration timepoint="0" setup="{setup}">'
        f'<ViewTransform type="affine"><Name>InterpolatedAffineModel3D</Name>'
        f"<affine>{xml_io.format_affine(stitch)}</affine></ViewTransform>"
        f'<ViewTransform type="affine"><Name>Translation to Nominal Grid</Name>'
        f"<affine>{xml_io.format_affine(nominal)}</affine></ViewTransform></ViewRegistration>"
    )


def make_dataset(folder: Path) -> Path:
    rng = np.random.default_rng(0)
    world = gaussian_filter(rng.normal(size=(64, STRIDE + TILE, STRIDE + TILE)), (1, 1.5, 1.5))
    world = ((world - world.min()) * 2000 + 100).astype(np.uint16)
    root = folder / "tiles"
    setups, groups, registrations = [], [], []
    for setup, (x, y) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        name = f"Tile_X_{x:04d}_Y_{y:04d}_Z_0000_ch_405.ome.zarr"
        data = world[:, y * STRIDE : y * STRIDE + TILE, x * STRIDE : x * STRIDE + TILE]
        group = zarr.open_group(str(root / name), mode="w")
        group.create_dataset("0", data=data[None, None], chunks=(1, 1, 32, 64, 64))
        group.attrs["multiscales"] = [
            {"datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1, 1, VOXEL_XYZ[2], VOXEL_XYZ[1], VOXEL_XYZ[0]]}
            ]}]}
        ]
        stitch = np.eye(4)
        stitch[:3, 3] = ERRORS_XYZ.get((x, y), 0)
        setups.append(
            f"<ViewSetup><id>{setup}</id><name>{name}</name><size>{TILE} {TILE} 64</size>"
            f"<voxelSize><unit>um</unit><size>{' '.join(map(str, VOXEL_XYZ))}</size></voxelSize></ViewSetup>"
        )
        groups.append(f'<zgroup setup="{setup}" tp="0" path="{name}" />')
        registrations.append(_registration(setup, stitch, (x * STRIDE, y * STRIDE, 0)))
    results = folder / "results"
    results.mkdir()
    (results / "bigstitcher.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><SpimData version="0.2"><SequenceDescription>'
        f'<ImageLoader format="bdv.multimg.zarr"><zarr type="absolute">{root.as_posix()}/</zarr>'
        f"<zgroups>{''.join(groups)}</zgroups></ImageLoader>"
        f"<ViewSetups>{''.join(setups)}</ViewSetups></SequenceDescription>"
        f"<ViewRegistrations>{''.join(registrations)}</ViewRegistrations></SpimData>",
        encoding="utf-8",
    )
    return results


def config(**guardrails) -> BlockAffineConfig:
    return BlockAffineConfig(
        pyramid_level=0,
        block_long_short_z=(48, 32, 24),
        guardrails=qc.Guardrails(**guardrails),
    )


@pytest.fixture()
def results(tmp_path):
    return make_dataset(tmp_path)


def test_off_mode_does_nothing(results):
    before = (results / "bigstitcher.xml").read_bytes()
    assert run("off", results, config()) is None
    assert sorted(path.name for path in results.iterdir()) == ["bigstitcher.xml"]
    assert (results / "bigstitcher.xml").read_bytes() == before


def test_apply_corrects_known_placement_error(results):
    seed = xml_io.read_mosaic_xml(results / "bigstitcher.xml")
    decision = run("apply", results, config(), workers=2)
    assert decision["passed"], decision["reasons"]
    assert decision["selected_placement"] == "candidate"
    assert decision["anchor_tile"] == "X0Y0"
    assert decision["holdout_median_relative_change_pct"] < -10
    refined = xml_io.read_mosaic_xml(results / "bigstitcher.xml")
    for label, tile in refined.tiles.items():
        x, y = xml_io.tile_xy(label)
        truth = np.eye(4)
        truth[:3, 3] = (x * STRIDE, y * STRIDE, 0)
        center = np.r_[np.asarray(tile.size_xyz) / 2, 1.0]
        error = (tile.matrix @ center - truth @ center)[:3]
        assert np.linalg.norm(error) < 0.3, (label, error)
        assert tile.transforms[0][0] == "InterpolatedAffineModel3D"
        np.testing.assert_allclose(tile.transforms[1][1], seed.tiles[label].transforms[1][1])
    assert (results / "bigstitcher_seed.xml").is_file()
    for name in ("decision.json", "training_blocks.csv", "holdout_seams.csv", "tile_guardrails.csv"):
        assert (results / "block_affine" / name).is_file()


def test_shadow_mode_keeps_production_xml(results):
    before = (results / "bigstitcher.xml").read_bytes()
    decision = run("shadow", results, config(), workers=2)
    assert decision["passed"]
    assert decision["selected_placement"] == "seed"
    assert (results / "bigstitcher.xml").read_bytes() == before
    assert (results / "block_affine" / "bigstitcher_block_affine.xml").is_file()


def test_apply_falls_back_when_guardrail_fails(results):
    before = (results / "bigstitcher.xml").read_bytes()
    decision = run("apply", results, config(max_center_displacement_um=0.01), workers=2)
    assert not decision["passed"]
    assert any("center displacement" in reason for reason in decision["reasons"])
    assert decision["selected_placement"] == "seed"
    assert (results / "bigstitcher.xml").read_bytes() == before


def test_apply_falls_back_when_images_are_missing(results, tmp_path):
    before = (results / "bigstitcher.xml").read_bytes()
    for path in (tmp_path / "tiles").iterdir():
        (path / ".zgroup").unlink()
    decision = run("apply", results, config(), log=lambda message: None)
    assert decision["selected_placement"] == "seed"
    assert not decision["passed"]
    assert (results / "bigstitcher.xml").read_bytes() == before
