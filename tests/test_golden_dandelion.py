"""Golden reproduction of the validated Dandelion R3 level-2 block-affine solution."""

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from block_affine import solve, xml_io

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def seed_xml(tmp_path_factory):
    path = tmp_path_factory.mktemp("golden") / "bigstitcher.xml"
    with gzip.open(DATA / "dandelion_r3_bigstitcher.xml.gz", "rb") as source, open(path, "wb") as target:
        shutil.copyfileobj(source, target)
    return path


@pytest.fixture(scope="module")
def expected():
    return json.loads((DATA / "dandelion_r3_l2_expected_solution.json").read_text())


@pytest.fixture(scope="module")
def solution(seed_xml):
    mosaic = xml_io.read_mosaic_xml(seed_xml)
    blocks = pd.read_csv(DATA / "dandelion_r3_l2_training_blocks.csv.gz")
    measurements = solve.build_measurements(blocks, mosaic)
    anchor = xml_io.center_tile(mosaic.labels)
    metadata = solve.build_metadata(measurements, mosaic.labels, anchor, mosaic.voxel_size_xyz)
    return mosaic, measurements, metadata, solve.solve(measurements, metadata)


def test_grid_seams_and_anchor_match_experiment(solution, expected):
    mosaic, _, metadata, _ = solution
    assert len(mosaic.labels) == 25
    assert len(xml_io.face_seams(mosaic.labels)) == 40
    assert metadata["anchor_tile"] == expected["anchor_tile"]


def test_reproduces_validated_solution(solution, expected):
    _, measurements, _, result = solution
    assert len(measurements) == expected["accepted_training_blocks"]
    assert result["ridge"] == expected["selected_affine_ridge"]
    for tile, matrix in expected["correction_matrices_voxel_xyz"].items():
        np.testing.assert_allclose(result["corrections_voxel"][tile], matrix, atol=1e-9)


def test_refined_xml_composes_correction_into_stitching_transform(solution, seed_xml, tmp_path):
    mosaic, _, _, result = solution
    output = tmp_path / "refined.xml"
    assert xml_io.write_refined_xml(seed_xml, output, result["corrections_voxel"]) == 25
    refined = xml_io.read_mosaic_xml(output)
    for label, tile in refined.tiles.items():
        seed_tile = mosaic.tiles[label]
        assert [name for name, _ in tile.transforms] == [name for name, _ in seed_tile.transforms]
        assert tile.transforms[0][0] == "InterpolatedAffineModel3D"
        np.testing.assert_allclose(tile.transforms[1][1], seed_tile.transforms[1][1])
        np.testing.assert_allclose(
            tile.matrix, result["corrections_voxel"][label] @ seed_tile.matrix, atol=1e-8
        )
