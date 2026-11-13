"""processing.json entry for the block-affine step (needs the capsule's aind-data-schema)."""

import json
import time
from datetime import datetime, timezone

import pytest

pytest.importorskip("aind_data_schema")
pytest.importorskip("psutil")

from aind_data_schema.components.identifiers import Code  # noqa: E402
from aind_data_schema.core.processing import DataProcess, Processing, ProcessStage  # noqa: E402
from aind_data_schema_models.process_names import ProcessName  # noqa: E402

from block_affine.metadata import PROCESS_NAME, data_process  # noqa: E402
from util import utils  # noqa: E402

DECISION = {
    "mode": "apply",
    "config": {"pyramid_level": 2, "guardrails": {"max_center_displacement_um": 10.0}},
    "passed": True,
    "reasons": [],
    "image_root": "s3://aind-open-data/HCR_X/image_radial_correction/",
    "selected_placement": "candidate",
    "accepted_training_blocks": 297,
    "holdout_median_relative_change_pct": -54.0,
    "traceback": "not recorded",
}


def _process(process_type: ProcessName, name: str = None) -> DataProcess:
    now = datetime.now(timezone.utc)
    return DataProcess(
        process_type=process_type,
        **({"name": name} if name else {}),
        stage=ProcessStage.PROCESSING,
        code=Code(url="https://example.org/repo"),
        experimenters=["Test"],
        start_date_time=now,
        end_date_time=now,
    )


def _entry(**kwargs):
    start = time.time()
    return data_process(
        DECISION, start, start + 1, repo_url="https://example.org/repo", version="1.0",
        commit_hash=None, experimenters=["Test"], **kwargs
    )


def test_entry_records_config_and_decision():
    entry = json.loads(_entry().model_dump_json())
    assert entry["name"] == PROCESS_NAME
    assert entry["code"]["parameters"]["mode"] == "apply"
    assert entry["code"]["parameters"]["input_location"] == DECISION["image_root"]
    assert entry["output_parameters"]["selected_placement"] == "candidate"
    assert "traceback" not in entry["output_parameters"]


def test_append_extends_dependency_graph_and_validates(tmp_path):
    # Mimics production: an existing history with a graph, then the BigStitcher entry
    # appended by generate_processing without a graph node.
    history = Processing.create_with_sequential_process_graph(
        data_processes=[_process(ProcessName.IMAGE_SPOT_DETECTION)]
    )
    processing = json.loads(history.model_dump_json())
    processing["data_processes"].append(
        json.loads(_process(ProcessName.IMAGE_TILE_ALIGNMENT).model_dump_json())
    )
    (tmp_path / "processing.json").write_text(json.dumps(processing))

    utils.append_to_processing(_entry(), tmp_path, "s3://bucket/key.json", upload=False)

    result = json.loads((tmp_path / "processing.json").read_text())
    names = [process["name"] for process in result["data_processes"]]
    assert names[-1] == PROCESS_NAME
    assert result["dependency_graph"][PROCESS_NAME] == ["Image tile alignment"]
    assert result["dependency_graph"]["Image tile alignment"] == ["Image spot detection"]
    Processing.model_validate(result)


def test_append_keeps_names_unique(tmp_path):
    processing = json.loads(
        Processing(data_processes=[_process(ProcessName.IMAGE_TILE_ALIGNMENT, PROCESS_NAME)]).model_dump_json()
    )
    (tmp_path / "processing.json").write_text(json.dumps(processing))
    utils.append_to_processing(_entry(), tmp_path, upload=False)
    names = [p["name"] for p in json.loads((tmp_path / "processing.json").read_text())["data_processes"]]
    assert names == [PROCESS_NAME, f"{PROCESS_NAME}_1"]
