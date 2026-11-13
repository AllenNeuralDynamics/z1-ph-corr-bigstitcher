"""Optional block-affine refinement of the BigStitcher channel-405 tile placement.

Modes:
    off     - not run; outputs are identical to the production capsule.
    shadow  - compute and QC a candidate; ``bigstitcher.xml`` is left unchanged.
    apply   - replace ``bigstitcher.xml`` with the candidate only if every QC gate
              passes; otherwise keep the BigStitcher seed.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from . import measure, qc, solve, xml_io

MODES = ("off", "shadow", "apply")
CANDIDATE_XML = "bigstitcher_block_affine.xml"
SEED_XML = "bigstitcher_seed.xml"


@dataclass
class BlockAffineConfig:
    pyramid_level: int = 2
    holdout_pyramid_level: Optional[int] = None  # defaults to pyramid_level
    training: measure.SamplingLayout = field(
        default_factory=lambda: measure.SamplingLayout((3, 3), 0.0, 0.25)
    )
    # Shifted 4 x 4 grid, spatially distinct from the 3 x 3 training blocks.
    holdout: measure.SamplingLayout = field(
        default_factory=lambda: measure.SamplingLayout((4, 4), 0.12, 0.12)
    )
    max_shift_xyz: Tuple[int, int, int] = (12, 12, 6)
    block_long_short_z: Tuple[int, int, int] = (768, 64, 128)
    min_peak_snr: float = 5.0
    min_correlation_after: float = 0.05
    affine_ridges: Tuple[float, ...] = solve.AFFINE_RIDGES
    fixed_ridge: Optional[float] = None
    anchor_tile: Optional[str] = None  # defaults to the grid-center tile
    guardrails: qc.Guardrails = field(default_factory=qc.Guardrails)
    s3_anonymous: bool = False

    def measurement(self, level: int) -> measure.MeasurementSettings:
        return measure.MeasurementSettings(
            pyramid_level=level,
            max_shift_xyz=tuple(self.max_shift_xyz),
            block_long_short_z=tuple(self.block_long_short_z),
            min_peak_snr=self.min_peak_snr,
            min_correlation_after=self.min_correlation_after,
            s3_anonymous=self.s3_anonymous,
        )

    def as_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=list))


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def refine(
    seed_xml: Path,
    output_dir: Path,
    config: BlockAffineConfig,
    workers: int = 1,
    open_array: Optional[measure.OpenLevel] = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Measure, solve and QC a candidate placement. Never modifies ``seed_xml``."""
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    mosaic = xml_io.read_mosaic_xml(seed_xml)
    labels = mosaic.labels
    seams = xml_io.face_seams(labels)
    anchor = config.anchor_tile or xml_io.center_tile(labels)
    holdout_level = (
        config.pyramid_level
        if config.holdout_pyramid_level is None
        else config.holdout_pyramid_level
    )
    log(
        f"[block-affine] {len(labels)} tiles, {len(seams)} face seams, anchor {anchor}, "
        f"level {config.pyramid_level} (holdout level {holdout_level}), workers {workers}"
    )
    if not seams:
        raise ValueError("No face-sharing tile seams; block affine needs a tile grid")

    training_blocks, training_seams = measure.measure_mosaic(
        mosaic, seams, config.training, config.measurement(config.pyramid_level),
        workers=workers, open_array=open_array, log=log,
    )
    training_blocks.to_csv(output_dir / "training_blocks.csv", index=False)
    training_seams.to_csv(output_dir / "training_seams.csv", index=False)
    log(f"[block-affine] training measured ({time.time() - started:.0f} s)")

    measurements = (
        solve.build_measurements(training_blocks, mosaic)
        if not training_blocks.empty
        else pd.DataFrame()
    )
    metadata = solve.build_metadata(measurements, labels, anchor, mosaic.voxel_size_xyz)
    measurements.to_csv(output_dir / "training_measurements.csv", index=False)
    solution = solve.solve(
        measurements, metadata, ridges=config.affine_ridges, fixed_ridge=config.fixed_ridge
    )
    solution["cross_validation"].to_csv(output_dir / "cross_validation.csv", index=False)
    solution["model_summary"].to_csv(output_dir / "model_summary.csv", index=False)
    candidate_xml = output_dir / CANDIDATE_XML
    xml_io.write_refined_xml(seed_xml, candidate_xml, solution["corrections_voxel"])
    _write_json(
        output_dir / "solution.json",
        {
            **metadata,
            "accepted_training_blocks": int(len(measurements)),
            "cross_validation": "three spatial folds: (z_index + long_index) modulo 3",
            "robust_fit": "Huber IRLS",
            "affine_ridge_candidates": list(config.affine_ridges),
            "fixed_ridge": config.fixed_ridge,
            "selected_affine_ridge": solution["ridge"],
            "parameters": solve.parameter_dict(
                solution["parameters"], solution["offsets"], metadata
            ),
            "correction_matrices_um": solution["corrections_um"],
            "correction_matrices_voxel_xyz": solution["corrections_voxel"],
            "seed_xml": str(seed_xml),
            "candidate_xml": candidate_xml.name,
        },
    )
    log(f"[block-affine] solved: ridge {solution['ridge']:g}, {len(measurements)} blocks")

    guardrail_table = qc.tile_guardrail_table(
        mosaic, solution["corrections_um"], measurements, config.guardrails
    )
    guardrail_table.to_csv(output_dir / "tile_guardrails.csv", index=False)
    disconnected = qc.connected_to_anchor(labels, measurements, anchor)

    holdout_settings = config.measurement(holdout_level)
    candidate_mosaic = xml_io.read_mosaic_xml(candidate_xml)
    holdout_tables = {}
    for name, placement in (("seed", mosaic), ("candidate", candidate_mosaic)):
        blocks, seam_table = measure.measure_mosaic(
            placement, seams, config.holdout, holdout_settings,
            workers=workers, open_array=open_array, log=log,
        )
        blocks.to_csv(output_dir / f"holdout_blocks_{name}.csv", index=False)
        holdout_tables[name] = (blocks, qc.seam_summary(blocks, seam_table))
    paired = qc.paired_holdout(
        holdout_tables["seed"][1],
        holdout_tables["candidate"][1],
        holdout_tables["seed"][0],
        holdout_tables["candidate"][0],
    )
    paired.to_csv(output_dir / "holdout_seams.csv", index=False)
    holdout = qc.holdout_decision(paired, config.guardrails)
    log(f"[block-affine] holdout measured ({time.time() - started:.0f} s)")

    reasons = list(holdout.pop("reasons"))
    failed_tiles = guardrail_table[~guardrail_table["passed"]]
    for row in failed_tiles.itertuples():
        reasons.append(f"tile {row.tile}: {row.failures}")
    if disconnected:
        reasons.append("tiles not connected to the anchor by accepted blocks: " + ", ".join(disconnected))
    failed_seams = training_seams[training_seams["error"].astype(str) != ""]
    return {
        "passed": not reasons,
        "reasons": reasons,
        "image_root": mosaic.image_root,
        "anchor_tile": anchor,
        "tiles": len(labels),
        "seams": len(seams),
        "training_failed_seams": failed_seams["seam"].tolist(),
        "accepted_training_blocks": int(len(measurements)),
        "total_training_blocks": int(len(training_blocks)),
        "selected_affine_ridge": solution["ridge"],
        "max_center_displacement_um": float(guardrail_table["center_displacement_um"].max()),
        "max_corner_displacement_um": float(guardrail_table["max_corner_displacement_um"].max()),
        **holdout,
        "runtime_s": round(time.time() - started, 1),
    }


def run(
    mode: str,
    results_folder: Path,
    config: BlockAffineConfig,
    workers: int = 1,
    log: Callable[[str], None] = print,
    open_array: Optional[measure.OpenLevel] = None,
) -> Optional[dict]:
    """Run the refinement next to ``results_folder/bigstitcher.xml``.

    Never raises in shadow/apply mode: any failure keeps the BigStitcher seed.
    """
    if mode not in MODES:
        raise ValueError(f"block_affine mode must be one of {MODES}; got {mode!r}")
    if mode == "off":
        return None
    production_xml = results_folder / "bigstitcher.xml"
    seed_xml = results_folder / SEED_XML
    output_dir = results_folder / "block_affine"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(production_xml, seed_xml)
    decision = {"mode": mode, "config": config.as_dict()}
    try:
        decision.update(refine(seed_xml, output_dir, config, workers, open_array, log))
    except Exception as error:  # noqa: BLE001 - fall back to the seed on any failure
        log(f"[block-affine] failed; keeping BigStitcher placement: {type(error).__name__}: {error}")
        decision.update(
            {"passed": False, "reasons": [f"{type(error).__name__}: {error}"],
             "traceback": traceback.format_exc()}
        )
    selected = "candidate" if mode == "apply" and decision["passed"] else "seed"
    if selected == "candidate":
        shutil.copy2(output_dir / CANDIDATE_XML, production_xml)
    decision["selected_placement"] = selected
    decision["output_xml"] = str(production_xml)
    _write_json(output_dir / "decision.json", decision)
    log(
        f"[block-affine] mode={mode} passed={decision['passed']} selected={selected}"
        + ("" if decision["passed"] else f" reasons={decision['reasons']}")
    )
    return decision
