"""AIND processing metadata for the block-affine refinement step."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from aind_data_schema.base import _GenericModel
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import DataProcess, ProcessStage
from aind_data_schema_models.process_names import ProcessName

PROCESS_NAME = "Block-affine tile refinement"
OUTPUT_KEYS = (
    "passed",
    "selected_placement",
    "reasons",
    "anchor_tile",
    "tiles",
    "seams",
    "training_failed_seams",
    "accepted_training_blocks",
    "total_training_blocks",
    "selected_affine_ridge",
    "max_center_displacement_um",
    "max_corner_displacement_um",
    "holdout_seams",
    "holdout_seed_median_shift_um",
    "holdout_candidate_median_shift_um",
    "holdout_median_relative_change_pct",
    "holdout_fraction_seams_improved",
    "degraded_seams",
)


def data_process(
    decision: dict,
    start_time: float,
    end_time: float,
    *,
    repo_url: str,
    version: str,
    commit_hash: Optional[str],
    experimenters: list,
) -> DataProcess:
    """Describe one block-affine run; its config goes in parameters, its decision in outputs."""
    image_root = decision.get("image_root", "")
    return DataProcess(
        process_type=ProcessName.IMAGE_TILE_ALIGNMENT,
        name=PROCESS_NAME,
        stage=ProcessStage.PROCESSING,
        code=Code(
            url=repo_url,
            name="z1-ph-corr-bigstitcher",
            version=version,
            commit_hash=commit_hash,
            run_script="/code/run",
            language="Python",
            parameters=_GenericModel(
                mode=decision["mode"],
                input_location=image_root,
                **decision.get("config", {}),
            ),
        ),
        experimenters=experimenters,
        start_date_time=datetime.fromtimestamp(start_time, timezone.utc),
        end_date_time=datetime.fromtimestamp(end_time, timezone.utc),
        output_path="block_affine",
        output_parameters=_GenericModel(
            **{key: decision[key] for key in OUTPUT_KEYS if key in decision}
        ),
        notes=(
            f"Block-affine refinement of the BigStitcher channel tile placement "
            f"(mode={decision['mode']}); selected placement: "
            f"{decision.get('selected_placement', 'seed')}. Tile images streamed from {image_root}."
        ),
    )
