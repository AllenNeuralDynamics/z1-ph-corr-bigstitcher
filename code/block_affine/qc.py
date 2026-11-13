"""Guardrails and held-out acceptance checks for a block-affine candidate."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .xml_io import MosaicXML


@dataclass(frozen=True)
class Guardrails:
    # Per-tile limits on the correction (not the full placement); about 2x the largest
    # values of the validated Dandelion/Chamomile R3 solutions (5.3 um center, 11.7 um
    # corner, 0.0085 scale deviation).
    max_center_displacement_um: float = 10.0
    max_corner_displacement_um: float = 25.0
    max_singular_value_deviation: float = 0.02
    max_determinant_deviation: float = 0.02
    min_accepted_blocks_per_tile: int = 3
    # Held-out seed-vs-candidate comparison.
    min_median_seam_improvement_pct: float = 10.0
    min_fraction_seams_improved: float = 0.5
    # A well-measured seam (at least this many held-out blocks accepted under both
    # placements) must not get materially worse on those common blocks.
    min_common_accepted_blocks: int = 3
    degraded_seam_abs_um: float = 0.3
    degraded_seam_rel_pct: float = 50.0
    max_degraded_seams: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _tile_corners_um(mosaic: MosaicXML, label: str) -> np.ndarray:
    tile = mosaic.tiles[label]
    size = tile.size_xyz.astype(float) - 1
    corners = np.array(
        [[x, y, z, 1.0] for x in (0, size[0]) for y in (0, size[1]) for z in (0, size[2])]
    )
    world_vox = (tile.matrix @ corners.T).T[:, :3]
    return world_vox * mosaic.voxel_size_xyz


def tile_guardrail_table(
    mosaic: MosaicXML,
    corrections_um: Dict[str, np.ndarray],
    measurements: pd.DataFrame,
    guardrails: Guardrails,
) -> pd.DataFrame:
    counts = pd.concat([measurements["tile_a"], measurements["tile_b"]]).value_counts()
    rows = []
    for label in mosaic.labels:
        correction = corrections_um.get(label, np.eye(4))
        corners = _tile_corners_um(mosaic, label)
        homogeneous = np.c_[corners, np.ones(len(corners))]
        displaced = (correction @ homogeneous.T).T[:, :3] - corners
        center = corners.mean(axis=0)
        center_displacement = correction[:3, :3] @ center + correction[:3, 3] - center
        linear = correction[:3, :3]
        singular = np.linalg.svd(linear, compute_uv=False)
        row = {
            "tile": label,
            "accepted_blocks": int(counts.get(label, 0)),
            "center_displacement_um": float(np.linalg.norm(center_displacement)),
            "max_corner_displacement_um": float(np.linalg.norm(displaced, axis=1).max()),
            "determinant": float(np.linalg.det(linear)),
            "min_singular_value": float(singular.min()),
            "max_singular_value": float(singular.max()),
        }
        reasons = []
        if row["accepted_blocks"] < guardrails.min_accepted_blocks_per_tile:
            reasons.append("too few accepted blocks")
        if row["center_displacement_um"] > guardrails.max_center_displacement_um:
            reasons.append("center displacement")
        if row["max_corner_displacement_um"] > guardrails.max_corner_displacement_um:
            reasons.append("corner displacement")
        if np.max(np.abs(singular - 1)) > guardrails.max_singular_value_deviation:
            reasons.append("scale/shear")
        if abs(row["determinant"] - 1) > guardrails.max_determinant_deviation:
            reasons.append("determinant")
        row["passed"] = not reasons
        row["failures"] = "; ".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows)


def connected_to_anchor(
    tiles: Sequence[str], measurements: pd.DataFrame, anchor: str
) -> List[str]:
    """Tiles with no path of accepted-block seams to the anchor."""
    neighbors: Dict[str, set] = {tile: set() for tile in tiles}
    for tile_a, tile_b in measurements[["tile_a", "tile_b"]].drop_duplicates().itertuples(
        index=False
    ):
        neighbors[tile_a].add(tile_b)
        neighbors[tile_b].add(tile_a)
    seen = {anchor}
    frontier = [anchor]
    while frontier:
        tile = frontier.pop()
        for neighbor in neighbors[tile] - seen:
            seen.add(neighbor)
            frontier.append(neighbor)
    return sorted(set(tiles) - seen)


def seam_summary(blocks: pd.DataFrame, seams: pd.DataFrame) -> pd.DataFrame:
    """Per-seam all-block and accepted-block residual summaries."""
    rows = []
    for seam in seams.itertuples():
        selected = blocks[blocks["seam"] == seam.seam] if not blocks.empty else blocks
        measured = selected.dropna(subset=["magnitude_um"]) if len(selected) else selected
        accepted = measured[measured["accepted"].astype(bool)] if len(measured) else measured
        rows.append(
            {
                "seam": seam.seam,
                "tile_a": seam.tile_a,
                "tile_b": seam.tile_b,
                "blocks": int(len(selected)),
                "accepted_blocks": int(len(accepted)),
                "accepted_fraction": float(len(accepted) / len(selected)) if len(selected) else 0.0,
                "all_block_median_shift_um": float(measured["magnitude_um"].median())
                if len(measured)
                else np.nan,
                "all_block_p90_shift_um": float(measured["magnitude_um"].quantile(0.9))
                if len(measured)
                else np.nan,
                "accepted_median_shift_um": float(accepted["magnitude_um"].median())
                if len(accepted)
                else np.nan,
                "median_correlation_before": float(measured["correlation_before"].median())
                if len(measured)
                else np.nan,
                "error": getattr(seam, "error", ""),
            }
        )
    return pd.DataFrame(rows)


def paired_holdout(
    seed: pd.DataFrame,
    candidate: pd.DataFrame,
    seed_blocks: pd.DataFrame,
    candidate_blocks: pd.DataFrame,
) -> pd.DataFrame:
    """Per-seam seed-vs-candidate comparison at identical held-out block locations."""
    paired = seed.merge(candidate, on=["seam", "tile_a", "tile_b"], suffixes=("_seed", "_candidate"))
    paired["delta_all_block_median_shift_um"] = (
        paired["all_block_median_shift_um_candidate"] - paired["all_block_median_shift_um_seed"]
    )
    paired["relative_change_pct"] = (
        100 * paired["delta_all_block_median_shift_um"] / paired["all_block_median_shift_um_seed"]
    )
    columns = ["seam", "block_label", "magnitude_um", "accepted"]
    if seed_blocks.empty or candidate_blocks.empty:
        blocks = pd.DataFrame(columns=["seam", "magnitude_um_seed", "magnitude_um_candidate"])
    else:
        blocks = seed_blocks[columns].merge(
            candidate_blocks[columns], on=["seam", "block_label"], suffixes=("_seed", "_candidate")
        )
        blocks = blocks[
            blocks["accepted_seed"].astype(bool) & blocks["accepted_candidate"].astype(bool)
        ]
    common = (
        blocks.groupby("seam")
        .agg(
            common_accepted_blocks=("magnitude_um_seed", "size"),
            common_median_shift_um_seed=("magnitude_um_seed", "median"),
            common_median_shift_um_candidate=("magnitude_um_candidate", "median"),
        )
        .reset_index()
    )
    paired = paired.merge(common, on="seam", how="left")
    paired["common_accepted_blocks"] = paired["common_accepted_blocks"].fillna(0).astype(int)
    paired["common_delta_um"] = (
        paired["common_median_shift_um_candidate"] - paired["common_median_shift_um_seed"]
    )
    return paired


def holdout_decision(paired: pd.DataFrame, guardrails: Guardrails) -> dict:
    usable = paired.dropna(
        subset=["all_block_median_shift_um_seed", "all_block_median_shift_um_candidate"]
    )
    well_measured = usable["common_accepted_blocks"] >= guardrails.min_common_accepted_blocks
    degraded = usable[
        well_measured
        & (usable["common_delta_um"] > guardrails.degraded_seam_abs_um)
        & (
            100 * usable["common_delta_um"] / usable["common_median_shift_um_seed"]
            > guardrails.degraded_seam_rel_pct
        )
    ]
    median_change = float(usable["relative_change_pct"].median()) if len(usable) else np.nan
    improved = float((usable["delta_all_block_median_shift_um"] < 0).mean()) if len(usable) else 0.0
    reasons = []
    if not len(usable):
        reasons.append("no held-out seams measured under both placements")
    else:
        if not median_change <= -guardrails.min_median_seam_improvement_pct:
            reasons.append(
                f"median held-out seam change {median_change:+.1f}% "
                f"(need <= -{guardrails.min_median_seam_improvement_pct:g}%)"
            )
        if improved < guardrails.min_fraction_seams_improved:
            reasons.append(f"only {improved:.0%} of held-out seams improved")
        if len(degraded) > guardrails.max_degraded_seams:
            reasons.append(
                "well-measured seams materially worse: " + ", ".join(degraded["seam"])
            )
    return {
        "holdout_seams": int(len(usable)),
        "holdout_median_relative_change_pct": median_change,
        "holdout_fraction_seams_improved": improved,
        "holdout_seed_median_shift_um": float(usable["all_block_median_shift_um_seed"].median())
        if len(usable)
        else np.nan,
        "holdout_candidate_median_shift_um": float(
            usable["all_block_median_shift_um_candidate"].median()
        )
        if len(usable)
        else np.nan,
        "degraded_seams": degraded["seam"].tolist(),
        "reasons": reasons,
    }
