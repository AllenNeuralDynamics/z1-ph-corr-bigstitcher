"""
Run capsule for BigStitcher stitching
"""

import csv
import os
import re
import boto3
from pathlib import Path
from urllib.parse import urlparse

from alignment import bigstitcher
from metrics.run_metrics import RunMetrics
from util import utils

def write_solver_removed_links_csv(results_folder: Path) -> None:
    """
    Extract ONLY iterative-solver dropped links from /results/logs.txt:
    
    Writes:
      /results/solver_removed_links.csv

    Columns:
      tp_a,a,tp_b,b,u,v,error
    where (u,v) is the undirected edge (min(a,b), max(a,b)).
    """
    log_path = results_folder / "logs.txt"
    if not log_path.exists():
        print("[POST] logs.txt not found; cannot parse removed links.")
        return

    txt = log_path.read_text(errors="replace")

    re_removed = re.compile(
        r"Removed link from\s+(\d+)-(\d+)\s+to\s+(\d+)-(\d+)\s+\(error=([0-9.eE+-]+)\)"
    )

    rows = []
    seen = set()  # (tp, u, v) de-dupe in case line appears twice

    for m in re_removed.finditer(txt):
        tp_a, a, tp_b, b, err = m.groups()
        tp_a = int(tp_a); a = int(a)
        tp_b = int(tp_b); b = int(b)
        err_f = float(err)

        u, v = (a, b) if a <= b else (b, a)
        key = (tp_a, u, v)
        if key in seen:
            continue
        seen.add(key)

        rows.append([tp_a, a, tp_b, b, u, v, err_f])

    # Sort biggest error first (helpful for quick inspection)
    rows.sort(key=lambda r: r[6], reverse=True)

    out_csv = results_folder / "solver_removed_links.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tp_a", "a", "tp_b", "b", "u", "v", "error"])
        w.writerows(rows)

    print(f"[POST] wrote: {out_csv} ({len(rows)} rows)")

def run():
    """Function that runs image stitching with BigStitcher"""
    data_folder = Path(os.path.abspath("../data"))
    results_folder = Path(os.path.abspath("../results"))

    # data_folder = Path(data_folder)

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = [
        f"{data_folder}/processing_manifest.json",
        f"{data_folder}/data_description.json",
        f"{data_folder}/acquisition.json",
        f"{data_folder}/processed/data_description.json",
        f"{data_folder}/radial_correction_parameters.json",
    ]

    missing_files = utils.validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(
            f"We miss the following files in the capsule input: {missing_files}"
        )

    pipeline_config, dataset_name, acquisition_dict = utils.get_data_config(
        data_folder=data_folder,
        processing_manifest_path="processing_manifest.json",
        data_description_path="data_description.json",
        acquisition_path="acquisition.json",
    )

    voxel_resolution = utils.get_resolution(acquisition_dict)
    stitching_channel = pipeline_config["pipeline_processing"]["stitching"]["channel"]

    processed_data_description = utils.read_json_as_dict(required_input_elements[3])
    radial_parameters = utils.read_json_as_dict(required_input_elements[4])

    processed_asset_name = processed_data_description.get("name", None)
    bucket_name = radial_parameters.get("bucket_name", None)

    if processed_asset_name is None or bucket_name is None:
        raise ValueError("Stitching requires S3 paths in Code Ocean at the moment.")

    path_to_data = f"s3://{bucket_name}/{processed_asset_name}/image_radial_correction"
    output_settings = f"s3://{bucket_name}/{processed_asset_name}/image_tile_alignment/bdv_settings.xml"

    stitching_channel_path = data_folder.joinpath(f"processed")

    output_json_file = results_folder.joinpath(f"{dataset_name}_tile_metadata.json")

    # Computing image transformations with bigtstitcher
    bigstitcher.main(
        path_to_data = path_to_data,
        output_settings = output_settings,
        acquisition_path = required_input_elements[2],
        channel_wavelength = stitching_channel,
        stitching_channel_path=stitching_channel_path,
        voxel_resolution=voxel_resolution,
        output_json_file=output_json_file,
        results_folder=results_folder,
        dataset_name=dataset_name,
        res_for_transforms=(0.76, 0.76, 3.4),
        scale_for_transforms=4
    )

    write_solver_removed_links_csv(results_folder)

    dropped_csv_path = results_folder / "solver_removed_links.csv"
    xml_path = results_folder / "bigstitcher.xml"

    if not xml_path.exists():
        raise FileNotFoundError(f"Expected BigStitcher XML was not found: {xml_path}")

    if not dropped_csv_path.exists():
        dropped_csv_path = None

    metrics_output_path = (
        f"s3://{bucket_name}/{processed_asset_name}/image_tile_alignment/alignment_metrics"
    )

    print(f"QC output will be written to: {metrics_output_path}")

    run_metrics = RunMetrics(
        dropped_csv_path=dropped_csv_path,
        xml_path=xml_path,
        output_path=metrics_output_path,
    )
    run_metrics.run_alignment_metrics()

if __name__ == "__main__":
    run()
