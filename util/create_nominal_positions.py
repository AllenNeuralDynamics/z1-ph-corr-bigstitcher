"""
Utility functions to create BigStitcher XML files from acquisition.json
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_tile_coordinates(filename: str) -> tuple:
    """
    Extract X, Y, Z coordinates from filename like 'Tile_X_0000_Y_0001_Z_0000_ch_488.ome.zarr'

    Parameters:
    ----------
    filename: str
        The filename to extract coordinates from.

    Returns:
    -------
        tuple: (x, y, z) coordinates as integers
    """
    pattern = r"Tile_X_(\d+)_Y_(\d+)_Z_(\d+)"
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return 0, 0, 0


def get_channel_from_filename(filename: str) -> str:
    """
    Extract channel from filename like 'Tile_X_0000_Y_0000_Z_0000_ch_488.ome.zarr'

    Parameters:
    ----------
    filename: str
        The filename to extract the channel from.

    Returns:
    -------
        str: Channel name extracted from the filename, or "unknown" if not found.
    """
    pattern = r"ch_(\d+)"
    match = re.search(pattern, filename)
    return match.group(1) if match else "unknown"


def get_unique_tiles(images: List[Dict]) -> List[str]:
    """
    Get unique tile names from images list

    Parameters:
    ----------
    images: List[Dict]
        List of image dictionaries containing file names.

    Returns:
    -------
        List[str]
        List of unique tile names, sorted alphabetically.
    """
    tiles = set()
    for image in images:
        filename = image.get("file_name", "")
        x, y, z = extract_tile_coordinates(filename)
        tile_name = f"Tile_X_{x:04d}_Y_{y:04d}_Z_{z:04d}"
        tiles.add(tile_name)
    return sorted(list(tiles))


def get_unique_channels(images: List[Dict]) -> List[str]:
    """
    Get unique channels from images list

    Parameters:
    ----------
    images: List[Dict]
        List of image dictionaries containing file names and channel names.

    Returns:
    -------
        List[str]
        List of unique channel names, sorted alphabetically.
    """
    channels = set()
    for image in images:
        channel = image.get(
            "channel_name", get_channel_from_filename(image.get("file_name", ""))
        )
        channels.add(channel)
    return sorted(list(channels))


def get_tile_translations(images: List[Dict]) -> Dict[str, List[float]]:
    """
    Get translation information for each unique tile in pixels.
    Converts from micrometers to pixels using the scale information.

    Parameters:
    ----------
    images: List[Dict]
        List of image dictionaries containing file names and transforms
        with scale and translation information.


    Returns:
    -------
        tile_translations: Dict[str, List[float]]

        Dict mapping tile names to [x, y, z] translations in pixels
    """
    tile_translations = {}

    for image in images:
        filename = image.get("file_name", "")
        if not filename:
            continue

        x, y, z = extract_tile_coordinates(filename)
        tile_name = f"Tile_X_{x:04d}_Y_{y:04d}_Z_{z:04d}"

        # Skip if we already have this tile
        if tile_name in tile_translations:
            continue

        # Extract scale and translation from transforms
        transforms = image.get("image_to_acquisition_transform", [])
        scale = [1.0, 1.0, 1.0]  # default pixel size in micrometers
        translation = [0.0, 0.0, 0.0]  # default translation in micrometers

        for transform in transforms:
            if transform.get("object_type") == "Scale":
                scale = transform.get("scale", [1.0, 1.0, 1.0])[:3]
            elif transform.get("object_type") == "Translation":
                translation = transform.get("translation", [0.0, 0.0, 0.0])[:3]

        # Convert translation from micrometers to pixels
        translation_pixels = [
            translation[0] / scale[0] if scale[0] != 0 else 0.0,
            translation[1] / scale[1] if scale[1] != 0 else 0.0,
            translation[2] / scale[2] if scale[2] != 0 else 0.0,
        ]

        tile_translations[tile_name] = translation_pixels

    return tile_translations


def create_bigstitcher_xml(
    acquisition_json_path: str,
    output_xml_path: str,
    zarr_base_path: str = ".",
    stitching_channel: str = None,
) -> None:
    """
    Create BigStitcher XML file from acquisition.json

    Parameters:
    -----------
    acquisition_json_path: str
        Path to the acquisition.json file
    output_xml_path: str
        Path where the XML file will be saved
    zarr_base_path: str
        Base path for zarr files (default: ".")
    """

    # Read acquisition.json
    with open(acquisition_json_path, "r") as f:
        acquisition_data = json.load(f)

    # Extract imaging configuration
    data_streams = acquisition_data.get("data_streams", [])
    if not data_streams:
        raise ValueError("No data streams found in acquisition.json")

    imaging_config = None
    for stream in data_streams:
        configurations = stream.get("configurations", [])
        for config in configurations:
            if config.get("object_type") == "Imaging config":
                imaging_config = config
                break
        if imaging_config:
            break

    if not imaging_config:
        raise ValueError("No imaging configuration found")

    images = imaging_config.get("images", [])
    if not images:
        raise ValueError("No images found in imaging configuration")

    if stitching_channel:
        images = [
            img
            for img in images
            if str(img.get("channel_name")) == str(stitching_channel)
        ]

    # Get pixel size from the first image's transform
    default_voxel_size = [1.0, 1.0, 1.0]  # default pixel size in micrometers
    if images:
        transforms = images[0].get("image_to_acquisition_transform", [])
        for transform in transforms:
            if transform.get("object_type") == "Scale":
                scale = transform.get("scale", default_voxel_size)
                pixel_size = scale[:3]  # Take first 3 values
                break

    # Get image dimensions from first image with dimensions
    default_dimensions = [1920, 1920, 100]  # default values
    image_size = None
    for image in images:
        dimensions = image.get("dimensions")
        if dimensions and dimensions.get("object_type") == "Scale":
            scale = dimensions.get("scale", default_dimensions)
            image_size = [int(s) for s in scale]
            break
        else:
            print(f"Warning: Image {image.get('file_name', 'unknown')} \
                  does not have valid dimensions, using default.")
            image_size = default_dimensions

    if not image_size:
        raise ValueError("Problem getting image size!")

    # Get unique tiles, channels, and tile translations
    unique_tiles = get_unique_tiles(images)
    unique_channels = get_unique_channels(images)
    tile_translations = get_tile_translations(images)

    # Create XML structure
    root = ET.Element("SpimData", version="0.2")

    # BasePath
    base_path = ET.SubElement(root, "BasePath", type="relative")
    base_path.text = "."

    # SequenceDescription
    seq_desc = ET.SubElement(root, "SequenceDescription")

    # ImageLoader
    image_loader = ET.SubElement(
        seq_desc, "ImageLoader", format="bdv.multimg.zarr", version="1.0"
    )
    zarr_elem = ET.SubElement(image_loader, "zarr", type="absolute")
    zarr_elem.text = str(zarr_base_path).rstrip("/") + "/"

    # ZGroups
    zgroups = ET.SubElement(image_loader, "zgroups")
    tile_id_map = {tile: idx for idx, tile in enumerate(unique_tiles)}

    # Create zgroups for each image
    for image in images:
        filename = image.get("file_name", "")
        if filename:
            x, y, z = extract_tile_coordinates(filename)
            tile_name = f"Tile_X_{x:04d}_Y_{y:04d}_Z_{z:04d}"

            zgroup = ET.SubElement(
                zgroups, "zgroup", setup=str(tile_id_map[tile_name]), timepoint="0"
            )
            path_elem = ET.SubElement(zgroup, "path")
            path_elem.text = filename

    # ViewSetups
    view_setups = ET.SubElement(seq_desc, "ViewSetups")

    # Create ViewSetup for each image
    setup_id = 0
    

    for image in images:
        filename = image.get("file_name", "")
        channel = image.get("channel_name", get_channel_from_filename(filename))

        if filename:
            x, y, z = extract_tile_coordinates(filename)
            tile_name = f"Tile_X_{x:04d}_Y_{y:04d}_Z_{z:04d}"

            view_setup = ET.SubElement(view_setups, "ViewSetup")

            # ID
            id_elem = ET.SubElement(view_setup, "id")
            id_elem.text = str(tile_id_map[tile_name]) #str(setup_id)

            # Name
            name_elem = ET.SubElement(view_setup, "name")
            name_elem.text = filename

            # Size
            size_elem = ET.SubElement(view_setup, "size")
            size_elem.text = f"{image_size[0]} {image_size[1]} {image_size[2]}"

            # VoxelSize
            voxel_size = ET.SubElement(view_setup, "voxelSize")
            unit_elem = ET.SubElement(voxel_size, "unit")
            unit_elem.text = "µm"
            size_elem_voxel = ET.SubElement(voxel_size, "size")
            size_elem_voxel.text = f"{pixel_size[0]} {pixel_size[1]} {pixel_size[2]}"

            # Attributes
            attributes = ET.SubElement(view_setup, "attributes")

            illum_elem = ET.SubElement(attributes, "illumination")
            illum_elem.text = "0"

            channel_elem = ET.SubElement(attributes, "channel")
            channel_elem.text = channel

            tile_elem = ET.SubElement(attributes, "tile")
            tile_elem.text = str(tile_id_map[tile_name])

            angle_elem = ET.SubElement(attributes, "angle")
            angle_elem.text = "0"

            setup_id += 1

    # Attributes sections
    # Illumination attributes
    illum_attr = ET.SubElement(view_setups, "Attributes", name="illumination")
    illumination = ET.SubElement(illum_attr, "Illumination")
    illum_id = ET.SubElement(illumination, "id")
    illum_id.text = "0"
    illum_name = ET.SubElement(illumination, "name")
    illum_name.text = "0"

    # Channel attributes
    channel_attr = ET.SubElement(view_setups, "Attributes", name="channel")
    for channel in unique_channels:
        channel_elem = ET.SubElement(channel_attr, "Channel")
        ch_id = ET.SubElement(channel_elem, "id")
        ch_id.text = channel
        ch_name = ET.SubElement(channel_elem, "name")
        ch_name.text = channel

    # Tile attributes
    tile_attr = ET.SubElement(view_setups, "Attributes", name="tile")
    for tile in unique_tiles:
        tile_elem = ET.SubElement(tile_attr, "Tile")
        tile_id = ET.SubElement(tile_elem, "id")
        tile_id.text = str(tile_id_map[tile])
        tile_name_elem = ET.SubElement(tile_elem, "name")
        tile_name_elem.text = tile

    # Angle attributes
    angle_attr = ET.SubElement(view_setups, "Attributes", name="angle")
    angle_elem = ET.SubElement(angle_attr, "Angle")
    angle_id = ET.SubElement(angle_elem, "id")
    angle_id.text = "0"
    angle_name_elem = ET.SubElement(angle_elem, "name")
    angle_name_elem.text = "0"

    # Timepoints
    timepoints = ET.SubElement(seq_desc, "Timepoints", type="pattern")
    int_pattern = ET.SubElement(timepoints, "integerpattern")
    int_pattern.text = "0"

    # MissingViews
    ET.SubElement(seq_desc, "MissingViews")

    # ViewRegistrations
    view_registrations = ET.SubElement(root, "ViewRegistrations")

    # Create ViewRegistration for each unique tile with proper tile translations
    for tile_id, tile_name in enumerate(unique_tiles):
        view_reg = ET.SubElement(
            view_registrations, "ViewRegistration", timepoint="0", setup=str(tile_id)
        )
        view_transform = ET.SubElement(view_reg, "ViewTransform", type="affine")
        name_elem = ET.SubElement(view_transform, "Name")
        name_elem.text = "Translation to Nominal Grid"

        # Get translation for this tile in pixels
        translation_pixels = tile_translations.get(tile_name, [0.0, 0.0, 0.0])

        # Create affine transformation matrix: 3x4 matrix flattened to 12 values
        # Format: [m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23]
        # Which represents: [1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz]
        affine_elem = ET.SubElement(view_transform, "affine")
        affine_elem.text = (
            f"1.0 0.0 0.0 {translation_pixels[0]:.5f} "
            f"0.0 1.0 0.0 {translation_pixels[1]:.5f} "
            f"0.0 0.0 1.0 {translation_pixels[2]:.5f}"
        )

    # Write XML file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)

    with open(output_xml_path, "wb") as f:
        f.write(b"<?xml version='1.0' encoding='utf-8'?>\n")
        tree.write(f, encoding="utf-8")


# Example usage function
def create_xml_from_acquisition(
    acquisition_json_path: str,
    output_xml_path: str,
    zarr_base_path: Optional[str] = None,
    stitching_channel: Optional[str] = None,
) -> None:
    """
    Convenience function to create XML from acquisition.json

    Parameters:
    -----------
    acquisition_json_path: str
        Path to acquisition.json file
    output_xml_path: str
        Output path for XML file
    zarr_base_path: str, optional
        Base path for zarr files. If None, uses parent directory of acquisition.json
    """
    if zarr_base_path is None:
        zarr_base_path = str(Path(acquisition_json_path).parent)

    create_bigstitcher_xml(
        acquisition_json_path=acquisition_json_path,
        output_xml_path=output_xml_path,
        zarr_base_path=zarr_base_path,
        stitching_channel=stitching_channel,
    )

    # print(f"BigStitcher XML file created: {output_xml_path}")


if __name__ == "__main__":
    # Example usage
    create_xml_from_acquisition(
        acquisition_json_path="/data/acquisition.json",
        output_xml_path="/results/test_dataset.xml",
        zarr_base_path="/data/HCR_768582_2025-06-01_00-00-00/SPIM",
        stitching_channel=488,
    )
