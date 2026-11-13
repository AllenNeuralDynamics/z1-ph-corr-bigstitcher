import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import shift as image_shift

from block_affine import qc, solve, xml_io
from block_affine.measure import axis_starts, phase_correlation_shift


def test_phase_correlation_recovers_shift_to_apply_to_moving():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(32, 40, 48)).astype(np.float32)
    applied = np.array([2.0, -3.0, 4.0])
    moving = image_shift(reference, applied, order=1, mode="wrap")
    recovered, peak_snr = phase_correlation_shift(reference, moving, max_shift_zyx=(6, 6, 6))
    np.testing.assert_allclose(recovered, -applied, atol=0.15)
    assert peak_snr > 5


def test_axis_starts_span_overlap_and_respect_edge_fraction():
    assert axis_starts(10, 110, 20, 3) == [10, 50, 90]
    assert axis_starts(10, 110, 20, 3, edge_fraction=0.25) == [30, 50, 70]
    with pytest.raises(ValueError):
        axis_starts(0, 10, 20, 1)


def test_face_seams_and_center_tile_handle_irregular_grids():
    labels = ["X0Y0", "X1Y0", "X2Y0", "X0Y1", "X2Y1"]
    assert xml_io.face_seams(labels) == [
        ("X0Y0", "X1Y0"), ("X0Y0", "X0Y1"), ("X1Y0", "X2Y0"), ("X2Y0", "X2Y1")
    ]
    assert xml_io.center_tile([f"X{x}Y{y}" for x in range(5) for y in range(4)]) == "X2Y1"


def test_write_refined_xml_rejects_unsolved_xml(tmp_path):
    source = tmp_path / "nominal.xml"
    source.write_text(
        """<SpimData><SequenceDescription><ImageLoader><zarr type="absolute">/tiles</zarr></ImageLoader>
<ViewSetups><ViewSetup><id>0</id><name>Tile_X_0000_Y_0000_Z_0000_ch_405.ome.zarr</name>
<size>4 4 4</size><voxelSize><size>1 1 1</size></voxelSize></ViewSetup></ViewSetups></SequenceDescription>
<ViewRegistrations><ViewRegistration timepoint="0" setup="0"><ViewTransform type="affine">
<Name>Translation to Nominal Grid</Name><affine>1 0 0 0 0 1 0 0 0 0 1 0</affine></ViewTransform>
</ViewRegistration></ViewRegistrations></SpimData>""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a stitching transform"):
        xml_io.write_refined_xml(source, tmp_path / "out.xml", {"X0Y0": np.eye(4)})


def test_translation_fit_recovers_displacement_with_configured_anchor():
    measurements = pd.DataFrame(
        {
            "tile_a": ["X1Y2"] * 3,
            "tile_b": ["X2Y2"] * 3,
            "global_x_um": [0.0, 10.0, 20.0],
            "global_y_um": [0.0, 10.0, 20.0],
            "global_z_um": [0.0, 10.0, 20.0],
            "target_dx_um": [1.0] * 3,
            "target_dy_um": [-2.0] * 3,
            "target_dz_um": [0.5] * 3,
        }
    )
    metadata = {
        "tiles": ["X1Y2", "X2Y2"],
        "anchor_tile": "X1Y2",
        "spatial_center_um": [10.0, 10.0, 10.0],
        "spatial_scale_um": 1000.0,
    }
    parameters, offsets = solve.fit(measurements, metadata, affine=False)
    assert set(offsets) == {"X2Y2"}
    estimated = solve.predict(measurements, metadata, parameters, offsets, affine=False)
    np.testing.assert_allclose(estimated, [[1.0, -2.0, 0.5]] * 3)


def test_connectivity_reports_isolated_tiles():
    measurements = pd.DataFrame({"tile_a": ["X0Y0"], "tile_b": ["X1Y0"]})
    assert qc.connected_to_anchor(["X0Y0", "X1Y0", "X2Y0"], measurements, "X0Y0") == ["X2Y0"]
