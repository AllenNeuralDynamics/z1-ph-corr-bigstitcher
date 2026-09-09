"""
Computes stitching transformations using
bigstitcher for SmartSPIM data structure
"""

import json
import math
import os
import subprocess
from pathlib import Path
from time import time
from typing import List, Optional, Tuple
import dask.array as da
from aind_data_schema.core.processing import DataProcess, ProcessName
from natsort import natsorted

from util.bdv_checkerboard_settings import CheckerboardSettingsBDV
from util import utils, create_nominal_positions


def validate_capsule_inputs(input_elements: List[str]) -> List[str]:
    """
    Validates input elemts for a capsule in
    Code Ocean.

    Parameters
    -----------
    input_elements: List[str]
        Input elements for the capsule. This
        could be sets of files or folders.

    Returns
    -----------
    List[str]
        List of missing files
    """

    missing_inputs = []
    for required_input_element in input_elements:
        required_input_element = Path(required_input_element)

        if not required_input_element.exists():
            missing_inputs.append(str(required_input_element))

    return missing_inputs


def get_data_config(
    data_folder: str,
    processing_manifest_path: str = "processing_manifest.json",
    data_description_path: str = "data_description.json",
    acquisition_path: str = "acquisition.json",
) -> Tuple:
    """
    Returns the first smartspim dataset found
    in the data folder

    Parameters
    -----------
    data_folder: str
        Path to the folder that contains the data

    processing_manifest_path: str
        Path for the processing manifest

    data_description_path: str
        Path for the data description

    Returns
    -----------
    Tuple[Dict, str]
        Dict: Empty dictionary if the path does not exist,
        dictionary with the data otherwise.

        Str: Empty string if the processing manifest
        was not found
    """

    # Returning first smartspim dataset found
    # Doing this because of Code Ocean, ideally we would have
    # a single dataset in the pipeline

    processing_manifest_path = Path(f"{data_folder}/{processing_manifest_path}")
    data_description_path = Path(f"{data_folder}/{data_description_path}")

    if not processing_manifest_path.exists():
        raise ValueError(f"Please, check processing manifest path: {processing_manifest_path}")

    if not data_description_path.exists():
        raise ValueError(f"Please, check data description path: {data_description_path}")

    derivatives_dict = utils.read_json_as_dict(str(processing_manifest_path))
    data_description_dict = utils.read_json_as_dict(str(data_description_path))
    acquisition_dict = utils.read_json_as_dict(f"{data_folder}/{acquisition_path}")

    smartspim_dataset = data_description_dict["name"]

    return derivatives_dict, smartspim_dataset, acquisition_dict


def create_tile_metadata(
    dataset_path: str, multiscale: str, cols: int, rows: int, xyz_resolution: List[float]
) -> dict:
    """
    Creates tile metadata for image stitching with BigStitcher wrapper

    Parameters
    ----------
    dataset_path: str
        Dataset path
    multiscale: str
        Multiscale used for stitching. It should be a string
        pointing to the multiscale. e.g., "0" for high resolution.
    cols: int
        Number of columns in the dataset.
    rows: int
        Number of rows in the dataset.
    xyz_resolution: List[float]
        Image resolution in xyz order.

    Returns
    -------
    Dict
        Dictionary with tile metadata useful for stitching
    """

    smartspim_to_tile_metadata = []

    # Setting origin as (0,0) tile
    origin = (cols[0], rows[0])

    for curr_col in cols:
        for curr_row in rows:
            # Zarr path -> needs to be relative
            zarr_path = dataset_path.joinpath(f"{curr_col}0_{curr_row}0.zarr")
            if not zarr_path.exists():
                zarr_path = dataset_path.joinpath(f"{curr_col}0_{curr_row}0.ome.zarr")

            # zarr_path = Path(f"../{zarr_path.relative_to(current_script_dir)}")
            # Image data
            img_data = da.from_zarr(Path(zarr_path).joinpath(multiscale))

            # um position relative to origin
            um_position = [(curr_col - origin[0]), (curr_row - origin[1]), 0]

            pixel_position = [i / j for i, j in zip(um_position, xyz_resolution)]

            smartspim_to_tile_metadata.append(
                {
                    "file": str(zarr_path),
                    "size": [img_data.shape[-1], img_data.shape[-2], img_data.shape[-3]],
                    "pixel_resolution": xyz_resolution,
                    "position": pixel_position,
                }
            )

    return smartspim_to_tile_metadata


def get_stitching_dict(specimen_id: str, dataset_xml_path: str, downsample: Optional[int] = 2) -> dict:
    """
    A function that writes a stitching dictioonary that will be used for
    creating a json file that gives parmaters to bigstitcher sittching run

    Parameters
    ----------
    specimen_id: str
        Specimen ID
    dataset_xml_path: str
        Path where the xml is located
    downsample: Optional[int] = 2
        Image multiscale used for stitching

    Returns
    -------
    dict
        Dictionary with the stitching parameters
        used for bigstitcher
    """
    # assert pathlib.Path(dataset_xml_path).exists()

    stitching_dict = {
        "session_id": str(specimen_id),
        "memgb": 100,
        "parallel": utils.get_code_ocean_cpu_limit(),
        "dataset_xml": str(dataset_xml_path),
        "do_phase_correlation": True,
        "do_detection": False,
        "do_registrations": False,
        "phase_correlation_params": {
            "downsample": downsample,
            "min_correlation": 0.6,
            "max_shift_in_x": 10,
            "max_shift_in_y": 10,
            "max_shift_in_z": 10,
        },
    }
    return stitching_dict

def get_estimated_downsample(
    voxel_resolution: List[float], phase_corr_res: Tuple[float] = (8.0, 8.0, 4.0)
) -> int:
    """
    Estimate the multiscale level (power-of-two downsampling) such that
    the resolution at that level is at least the phase_corr_res in all axes.

    Parameters
    ----------
    voxel_resolution : List[float]
        Resolution of the original image at level 0 (in XYZ order).
    phase_corr_res : Tuple[float]
        Target resolution for phase correlation (in XYZ order).

    Returns
    -------
    int
        Estimated downsample level (0 or higher).
    """

    levels = []
    for vres, cres in zip(voxel_resolution, phase_corr_res):
        if cres < vres:
            raise ValueError(
                "phase_corr_res must be greater than or equal to voxel_resolution."
            )
        ratio = cres / vres
        levels.append(math.floor(math.log2(ratio)))

    return max(levels)

def get_max_shifts(
    shape: tuple, overlap: float, pyramid_level: int, min_shift: int = 10, room: int = 10
):
    """
    Calculate the maximum shifts in Z, Y, and X dimensions
    based on image shape and overlap percentage.

    Parameters
    ----------
    shape : tuple of int
        Shape of the image (Z, Y, X) at the given pyramid level.
    overlap : float
        Overlap as a fraction (e.g., 0.1 for 10%).
    pyramid_level : int
        Pyramid level from the zarr multiscale (1 = full res).
    min_shift : int
        Minimum shift allowed.
    room : int
        Extra tolerance to add.

    Returns
    -------
    tuple of int
        Maximum shift in (Z, Y, X) directions.
    """
    if not (0 <= overlap <= 1):
        raise ValueError("Overlap must be between 0 and 1.")

    # Ensure shape corresponds to this pyramid level
    level_shape = tuple(int(dim // (2 ** (pyramid_level - 1))) for dim in shape)

    shifts = []
    for dim in level_shape:
        s = int(dim * overlap)
        if s < min_shift:
            s = min_shift
        s += room
        shifts.append(s)

    return tuple(shifts)

def main(
    path_to_data,
    output_settings,
    acquisition_path,
    channel_wavelength,
    stitching_channel_path,
    voxel_resolution,
    output_json_file,
    results_folder,
    dataset_name,
    res_for_transforms=(0.19, 0.19, 0.85),
    scale_for_transforms=None
):
    """
    Computes image stitching with BigStitcher using Phase Correlation

    Parameters
    ----------
    stitching_channel_path: str
        Path where the stitching channel is located
    voxel_resolution: Tuple[float]
        Voxel resolution in order XYZ
    output_json_file: str
        Path where the json file will be written
    results_folder: Path
        Results folder
    smartspim_dataset_name: str
        SmartSPIM dataset name
    """

    BIGSTITCHER_PATH = os.getenv("BIGSTITCHER_HOME")

    if BIGSTITCHER_PATH is None:
        raise ValueError("Please, set the BIGSTITCHER_HOME env value.")

    BIGSTITCHER_PATH = Path(BIGSTITCHER_PATH)
    env = os.environ.copy()

    if not BIGSTITCHER_PATH.exists():
        raise ValueError("Please, set the BIGSTITCHER_PATH env value.")

    start_time = time()
    metadata_folder = results_folder.joinpath("metadata")
    utils.create_folder(str(metadata_folder))

    output_big_stitcher_xml = f"{results_folder}/bigstitcher.xml"

    # Creating XML
    create_nominal_positions.create_xml_from_acquisition(
        acquisition_json_path=acquisition_path,
        output_xml_path=output_big_stitcher_xml,
        zarr_base_path=path_to_data,
        stitching_channel=channel_wavelength,
    )

    if scale_for_transforms is None:
        scale_for_transforms = get_estimated_downsample(
            voxel_resolution=voxel_resolution, phase_corr_res=res_for_transforms
        )
    
    downsampled_scale = int(scale_for_transforms)

    is_proteomics = False
    project_name = utils.get_project_name()
    if project_name == "PLACE": 
        is_proteomics = True
        print("PROTEOMICS DATASET")

    max_shift = 160 // (downsampled_scale + 1)

    curr_folder = Path(os.path.realpath(__file__)).parent
    code_folder = curr_folder.parent
    run_classes_script = code_folder / "alignment/run_classes.sh"

    # Assuming machine with 128G and 16 cores
    env.update(
        {
            "JAVA_HEAP_SIZE": "128g",
            "SPARK_THREADS": "16",
            "MAIN_CLASS": "net.preibisch.bigstitcher.spark.SparkPairwiseStitching",
        }
    )

    if is_proteomics:
        stitching_command = [
            "bash",
            str(run_classes_script),
            "--xml",
            str(output_big_stitcher_xml),
            "--downsampling",
            f"{downsampled_scale},{downsampled_scale},{downsampled_scale}",
            "--minR",
            str(0.0)
        ]
    else:
        stitching_command = [
            "bash",
            str(run_classes_script),
            "--xml",
            str(output_big_stitcher_xml),
            "--downsampling",
            f"{downsampled_scale},{downsampled_scale},{downsampled_scale}",
            "--maxShiftZ",
            str(max_shift),
            "--maxShiftY",
            str(max_shift),
            "--maxShiftX",
            str(max_shift),
            "--minR",
            str(0.6)
        ]

    _ = subprocess.run(
        stitching_command,
        check=True,
        cwd=code_folder,
        env=env,
    )

    # Updating java class to solver for global optimization
    env.update({"MAIN_CLASS": "net.preibisch.bigstitcher.spark.Solver"})

    if is_proteomics:
        global_opt_command = [
            "bash",
            str(run_classes_script),
            "--xml",
            str(output_big_stitcher_xml),
            "-s",
            "STITCHING",
            "--method",
            "TWO_ROUND_ITERATIVE",
            "--maxError",
            "3",
            "--maxIterations",
            "10000",
            "--maxPlateauwidth",
            "200",
            "--relativeThreshold",
            "2.5",
            "--absoluteThreshold",
            "3.5",
        ]
    else:
        global_opt_command = [
            "bash",
            str(run_classes_script),
            "--xml",
            str(output_big_stitcher_xml),
            "--sourcePoints",
            "STITCHING",
        ]

    _ = subprocess.run(
        global_opt_command,
        check=True,
        cwd=code_folder,
        env=env,
    )
   
    end_time = time()

    output_big_stitcher_json = (
        f"{results_folder}/{dataset_name}_stitch_channel_{channel_wavelength}_params.json"
    )

    data_processes = []
    data_processes.append(
        DataProcess(
            name=ProcessName.IMAGE_TILE_ALIGNMENT,
            software_version="e112363",
            start_date_time=start_time,
            end_date_time=end_time,
            input_location=str(dataset_name),
            output_location=str(output_big_stitcher_json),
            outputs={"output_file": str(output_big_stitcher_json)},
            code_url="",
            code_version="1.2.7",
            parameters={"stitching": stitching_command, "global_optimization": global_opt_command},
            notes="Running stitching and global optimization separately",
        )
    )

    utils.generate_processing(
        data_processes=data_processes,
        dest_processing=metadata_folder,
        processor_full_name="Sean Fite",
        pipeline_version="3.0.0",
    )

    # Generate settings script for BDV
    checkerboard_settings = CheckerboardSettingsBDV(output_big_stitcher_xml, output_settings)
    checkerboard_settings.run()


if __name__ == "__main__":
    main()
