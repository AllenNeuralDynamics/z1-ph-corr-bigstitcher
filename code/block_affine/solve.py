"""Huber-robust, ridge-regularized near-identity affine correction per tile.

Ported from the RhaPSO stitching experiment (Code Ocean capsule 9066422,
``code/global_block_affine.py``); the model and fitting are unchanged.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .xml_io import MosaicXML, tile_xy

SPATIAL_SCALE_UM = 1000.0
# Keep affine candidates strictly regularized: on a finite grid an unregularized
# per-tile affine can extrapolate wildly at perimeter tiles while still scoring
# well on held-out blocks sampled from the same narrow overlap planes.
AFFINE_RIDGES = (0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0, 100.0)
TARGET_COLUMNS = ["target_dx_um", "target_dy_um", "target_dz_um"]
POSITION_COLUMNS = ["global_x_um", "global_y_um", "global_z_um"]


def build_measurements(
    blocks: pd.DataFrame, mosaic: MosaicXML
) -> pd.DataFrame:
    """Convert accepted seam blocks to world-micrometer positions and target shifts."""
    voxel = np.asarray(mosaic.voxel_size_xyz, dtype=float)
    rows = []
    accepted = blocks[blocks["accepted"].astype(bool)]
    for block in accepted.itertuples():
        transform_b = mosaic.tiles[block.tile_b].matrix
        block_size = np.array([block.block_x_vox, block.block_y_vox, block.block_z_vox])
        local_b = np.array([block.x_start, block.y_start, block.z_start]) + block_size / 2
        global_vox = (transform_b @ np.r_[local_b, 1.0])[:3]
        shift_global_vox = transform_b[:3, :3] @ np.array(
            [block.dx_vox, block.dy_vox, block.dz_vox], dtype=float
        )
        rows.append(
            {
                "seam": block.seam,
                "tile_a": block.tile_a,
                "tile_b": block.tile_b,
                "block_label": block.block_label,
                "z_index": int(block.z_index),
                "long_index": int(block.long_index),
                "fold": int((block.z_index + block.long_index) % 3),
                "global_x_um": global_vox[0] * voxel[0],
                "global_y_um": global_vox[1] * voxel[1],
                "global_z_um": global_vox[2] * voxel[2],
                "target_dx_um": shift_global_vox[0] * voxel[0],
                "target_dy_um": shift_global_vox[1] * voxel[1],
                "target_dz_um": shift_global_vox[2] * voxel[2],
                "phase_peak_snr": block.phase_peak_snr,
                "correlation_before": block.correlation_before,
            }
        )
    return pd.DataFrame(rows)


def build_metadata(
    measurements: pd.DataFrame, tiles: Sequence[str], anchor_tile: str, voxel_size_xyz
) -> dict:
    if measurements.empty:
        raise ValueError("No accepted phase-correlation blocks")
    if anchor_tile not in tiles:
        raise ValueError(f"Anchor tile {anchor_tile} is absent from the measured graph")
    return {
        "tiles": sorted(tiles),
        "anchor_tile": anchor_tile,
        "seams": int(measurements["seam"].nunique()),
        "voxel_size_xyz_um": [float(value) for value in voxel_size_xyz],
        "spatial_center_um": measurements[POSITION_COLUMNS].mean().tolist(),
        "spatial_scale_um": SPATIAL_SCALE_UM,
    }


def _design(
    measurements: pd.DataFrame, metadata: dict, *, affine: bool
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    width = 12 if affine else 3
    anchor_tile = metadata["anchor_tile"]
    fitted_tiles = [tile for tile in metadata["tiles"] if tile != anchor_tile]
    offsets = {tile: index * width for index, tile in enumerate(fitted_tiles)}
    design = np.zeros((3 * len(measurements), width * len(fitted_tiles)))
    target = measurements[TARGET_COLUMNS].to_numpy().reshape(-1)
    center = np.asarray(metadata["spatial_center_um"])
    positions = measurements[POSITION_COLUMNS].to_numpy()
    normalized = (positions - center) / metadata["spatial_scale_um"]
    for block_index, row in enumerate(measurements.itertuples()):
        basis = np.eye(3)
        if affine:
            basis = np.hstack(
                (np.eye(3), *(value * np.eye(3) for value in normalized[block_index]))
            )
        rows = slice(3 * block_index, 3 * block_index + 3)
        if row.tile_a != anchor_tile:
            start = offsets[row.tile_a]
            design[rows, start : start + width] += basis
        if row.tile_b != anchor_tile:
            start = offsets[row.tile_b]
            design[rows, start : start + width] -= basis
    return design, target, offsets


def _regularization(
    metadata: dict, offsets: Dict[str, int], affine_ridge: float
) -> np.ndarray:
    if affine_ridge <= 0:
        return np.zeros((0, 12 * len(offsets)))
    rows = []
    strength = np.sqrt(affine_ridge)
    anchor_tile = metadata["anchor_tile"]
    for offset in offsets.values():
        for parameter in range(3, 12):
            row = np.zeros(12 * len(offsets))
            row[offset + parameter] = strength
            rows.append(row)
    # Encourage neighboring tiles to have similar linear corrections.
    for tile_a in metadata["tiles"]:
        xy_a = tile_xy(tile_a)
        for tile_b in metadata["tiles"]:
            if tile_a >= tile_b:
                continue
            xy_b = tile_xy(tile_b)
            if abs(xy_a[0] - xy_b[0]) + abs(xy_a[1] - xy_b[1]) != 1:
                continue
            for parameter in range(3, 12):
                row = np.zeros(12 * len(offsets))
                if tile_a != anchor_tile:
                    row[offsets[tile_a] + parameter] += strength
                if tile_b != anchor_tile:
                    row[offsets[tile_b] + parameter] -= strength
                rows.append(row)
    return np.asarray(rows)


def fit(
    measurements: pd.DataFrame,
    metadata: dict,
    *,
    affine: bool,
    affine_ridge: float = 0,
) -> Tuple[np.ndarray, Dict[str, int]]:
    design, target, offsets = _design(measurements, metadata, affine=affine)
    regularization = (
        _regularization(metadata, offsets, affine_ridge)
        if affine
        else np.zeros((0, design.shape[1]))
    )
    parameters = np.linalg.lstsq(
        np.vstack((design, regularization)),
        np.r_[target, np.zeros(len(regularization))],
        rcond=None,
    )[0]
    for _ in range(8):
        residual = (target - design @ parameters).reshape(-1, 3)
        norms = np.linalg.norm(residual, axis=1)
        scale = 1.4826 * np.median(np.abs(norms - np.median(norms)))
        if scale < 1e-8:
            break
        weights = np.minimum(1.0, 1.5 * scale / np.maximum(norms, 1e-12))
        row_weights = np.repeat(np.sqrt(weights), 3)
        parameters = np.linalg.lstsq(
            np.vstack((design * row_weights[:, None], regularization)),
            np.r_[target * row_weights, np.zeros(len(regularization))],
            rcond=None,
        )[0]
    return parameters, offsets


def predict(
    measurements: pd.DataFrame,
    metadata: dict,
    parameters: np.ndarray,
    offsets: Dict[str, int],
    *,
    affine: bool,
) -> np.ndarray:
    design, _, expected_offsets = _design(measurements, metadata, affine=affine)
    if offsets != expected_offsets:
        raise ValueError("Tile ordering changed")
    return (design @ parameters).reshape(-1, 3)


def cross_validate(
    measurements: pd.DataFrame, metadata: dict, ridges: Sequence[float] = AFFINE_RIDGES
) -> pd.DataFrame:
    """Three spatial folds: (z_index + long_index) modulo 3."""
    rows = []
    candidates = [("translation", False, 0.0)] + [
        ("regularized_affine", True, float(ridge)) for ridge in ridges
    ]
    for name, affine, ridge in candidates:
        for fold in sorted(measurements["fold"].unique()):
            train = measurements[measurements["fold"] != fold]
            test = measurements[measurements["fold"] == fold]
            parameters, offsets = fit(train, metadata, affine=affine, affine_ridge=ridge)
            predicted = predict(test, metadata, parameters, offsets, affine=affine)
            residual = test[TARGET_COLUMNS].to_numpy() - predicted
            for source, estimate, remaining in zip(test.itertuples(), predicted, residual):
                rows.append(
                    {
                        "model": name,
                        "affine_ridge": ridge,
                        "fold": fold,
                        "seam": source.seam,
                        "block_label": source.block_label,
                        "target_magnitude_um": float(
                            np.linalg.norm(
                                [source.target_dx_um, source.target_dy_um, source.target_dz_um]
                            )
                        ),
                        "predicted_magnitude_um": float(np.linalg.norm(estimate)),
                        "residual_dx_um": remaining[0],
                        "residual_dy_um": remaining[1],
                        "residual_dz_um": remaining[2],
                        "residual_magnitude_um": float(np.linalg.norm(remaining)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_cross_validation(validation: pd.DataFrame) -> pd.DataFrame:
    return (
        validation.groupby(["model", "affine_ridge"])
        .agg(
            blocks=("block_label", "size"),
            median_residual_um=("residual_magnitude_um", "median"),
            mean_residual_um=("residual_magnitude_um", "mean"),
            p90_residual_um=("residual_magnitude_um", lambda x: x.quantile(0.9)),
        )
        .reset_index()
    )


def select_ridge(summary: pd.DataFrame) -> float:
    affine = summary[summary["model"] == "regularized_affine"].sort_values(
        ["mean_residual_um", "p90_residual_um", "median_residual_um"]
    )
    return float(affine.iloc[0]["affine_ridge"])


def parameter_dict(
    parameters: np.ndarray, offsets: Dict[str, int], metadata: dict
) -> Dict[str, dict]:
    result = {}
    for tile in metadata["tiles"]:
        values = np.zeros(12)
        if tile != metadata["anchor_tile"]:
            values = parameters[offsets[tile] : offsets[tile] + 12]
        result[tile] = {
            "translation_um": values[:3].tolist(),
            "linear_displacement_um_per_1000um": values[3:].reshape(3, 3, order="F").tolist(),
        }
    return result


def correction_matrices_um(
    parameters: np.ndarray, offsets: Dict[str, int], metadata: dict
) -> Dict[str, np.ndarray]:
    """Per-tile corrections acting on world micrometers."""
    center = np.asarray(metadata["spatial_center_um"])
    matrices = {}
    for tile in metadata["tiles"]:
        values = np.zeros(12)
        if tile != metadata["anchor_tile"]:
            values = parameters[offsets[tile] : offsets[tile] + 12]
        gradients = (
            np.column_stack((values[3:6], values[6:9], values[9:12])) / SPATIAL_SCALE_UM
        )
        correction = np.eye(4)
        correction[:3, :3] += gradients
        correction[:3, 3] = values[:3] - gradients @ center
        matrices[tile] = correction
    return matrices


def to_voxel_frame(
    corrections_um: Dict[str, np.ndarray], voxel_size_xyz
) -> Dict[str, np.ndarray]:
    """Express micrometer-frame corrections in the XML's world-voxel frame."""
    physical_to_voxel = np.eye(4)
    physical_to_voxel[:3, :3] = np.diag(1 / np.asarray(voxel_size_xyz, dtype=float))
    voxel_to_physical = np.linalg.inv(physical_to_voxel)
    return {
        tile: physical_to_voxel @ matrix @ voxel_to_physical
        for tile, matrix in corrections_um.items()
    }


def solve(
    measurements: pd.DataFrame,
    metadata: dict,
    ridges: Sequence[float] = AFFINE_RIDGES,
    fixed_ridge: Optional[float] = None,
) -> dict:
    """Cross-validate (unless a ridge is fixed), then fit on all accepted blocks."""
    validation = cross_validate(measurements, metadata, ridges)
    summary = summarize_cross_validation(validation)
    ridge = float(fixed_ridge) if fixed_ridge is not None else select_ridge(summary)
    parameters, offsets = fit(measurements, metadata, affine=True, affine_ridge=ridge)
    corrections_um = correction_matrices_um(parameters, offsets, metadata)
    return {
        "ridge": ridge,
        "parameters": parameters,
        "offsets": offsets,
        "cross_validation": validation,
        "model_summary": summary,
        "corrections_um": corrections_um,
        "corrections_voxel": to_voxel_frame(corrections_um, metadata["voxel_size_xyz_um"]),
    }
