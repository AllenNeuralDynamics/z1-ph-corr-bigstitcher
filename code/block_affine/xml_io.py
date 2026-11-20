"""BigDataViewer XML helpers for the block-affine refinement."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

TILE_PATTERN = re.compile(r"Tile_X_(\d{4})_Y_(\d{4})_")
# The downstream combine step only transfers index-0 transforms with these names.
STITCHING_TRANSFORM_MARKERS = ("AffineModel3D", "RigidModel3D", "Stitching Transform")


def parse_affine(text: str | None) -> np.ndarray:
    values = np.array([float(value) for value in (text or "").split()])
    if values.size != 12:
        raise ValueError(f"Expected a 3 x 4 affine; got {values.size} values")
    matrix = np.eye(4)
    matrix[:3] = values.reshape(3, 4)
    return matrix


def format_affine(matrix: np.ndarray) -> str:
    return " ".join(f"{value:.16g}" for value in np.asarray(matrix)[:3].ravel())


def tile_label(name: str) -> str | None:
    match = TILE_PATTERN.search(name)
    if match is None:
        return None
    return f"X{int(match.group(1))}Y{int(match.group(2))}"


def tile_xy(label: str) -> Tuple[int, int]:
    match = re.fullmatch(r"X(\d+)Y(\d+)", label)
    if match is None:
        raise ValueError(f"Not a tile label: {label}")
    return int(match.group(1)), int(match.group(2))


@dataclass
class TileInfo:
    label: str
    setup: int
    name: str
    image_uri: str
    size_xyz: np.ndarray
    transforms: List[Tuple[str, np.ndarray]] = field(default_factory=list)

    @property
    def matrix(self) -> np.ndarray:
        """Full voxel-to-world transform (first listed transform is applied last)."""
        result = np.eye(4)
        for _, transform in self.transforms:
            result = result @ transform
        return result


@dataclass
class MosaicXML:
    path: Path
    image_root: str
    voxel_size_xyz: np.ndarray
    tiles: Dict[str, TileInfo]

    @property
    def labels(self) -> List[str]:
        return sorted(self.tiles, key=tile_xy)


def read_mosaic_xml(path: Path) -> MosaicXML:
    """Parse a single-channel, single-timepoint BigStitcher XML."""
    path = Path(path)
    root = ET.parse(path).getroot()
    loader = root.find(".//SequenceDescription/ImageLoader")
    if loader is None:
        raise ValueError(f"{path} has no ImageLoader")
    image_root = (loader.findtext("zarr") or "").strip()
    if not image_root:
        raise ValueError(f"{path} ImageLoader has no zarr root")
    zgroups = {
        int(group.get("setup")): group.get("path")
        for group in loader.findall("zgroups/zgroup")
    }
    tiles: Dict[str, TileInfo] = {}
    voxel_sizes = []
    for setup in root.findall(".//ViewSetups/ViewSetup"):
        setup_id = int(setup.findtext("id"))
        name = setup.findtext("name") or ""
        label = tile_label(name)
        if label is None:
            raise ValueError(f"Cannot parse tile position from setup name {name!r}")
        if label in tiles:
            raise ValueError(
                f"Tile {label} appears more than once; expected a single-channel XML"
            )
        voxel_sizes.append(
            np.array([float(value) for value in setup.findtext("voxelSize/size").split()])
        )
        group = zgroups.get(setup_id, name)
        tiles[label] = TileInfo(
            label=label,
            setup=setup_id,
            name=name,
            image_uri=f"{image_root.rstrip('/')}/{group}",
            size_xyz=np.array([int(value) for value in setup.findtext("size").split()]),
        )
    if not tiles:
        raise ValueError(f"{path} has no tile view setups")
    voxel_size_xyz = voxel_sizes[0]
    if any(not np.allclose(size, voxel_size_xyz) for size in voxel_sizes[1:]):
        raise ValueError("All tile setups must use the same voxel size")
    by_setup = {tile.setup: tile for tile in tiles.values()}
    timepoints = set()
    for registration in root.findall(".//ViewRegistrations/ViewRegistration"):
        timepoints.add(registration.get("timepoint", "0"))
        tile = by_setup.get(int(registration.get("setup")))
        if tile is None:
            continue
        tile.transforms = [
            (transform.findtext("Name") or "", parse_affine(transform.findtext("affine")))
            for transform in registration.findall("ViewTransform")
        ]
    if len(timepoints) > 1:
        raise ValueError("Block-affine refinement supports a single timepoint")
    missing = [label for label, tile in tiles.items() if not tile.transforms]
    if missing:
        raise ValueError(f"Tiles without registrations: {missing}")
    return MosaicXML(
        path=path, image_root=image_root, voxel_size_xyz=voxel_size_xyz, tiles=tiles
    )


def face_seams(labels: List[str]) -> List[Tuple[str, str]]:
    """Face-sharing tile pairs, ordered (lower, higher) along the grid axis."""
    present = set(labels)
    seams = []
    for label in sorted(labels, key=tile_xy):
        x, y = tile_xy(label)
        for neighbor in (f"X{x + 1}Y{y}", f"X{x}Y{y + 1}"):
            if neighbor in present:
                seams.append((label, neighbor))
    return seams


def center_tile(labels: List[str]) -> str:
    """Tile closest to the grid center (ties broken by grid order)."""
    coordinates = np.array([tile_xy(label) for label in labels], dtype=float)
    center = (coordinates.min(axis=0) + coordinates.max(axis=0)) / 2
    distances = np.abs(coordinates - center).sum(axis=1)
    order = sorted(range(len(labels)), key=lambda i: (distances[i], tile_xy(labels[i])))
    return labels[order[0]]


def write_refined_xml(
    source_xml: Path, output_xml: Path, corrections: Dict[str, np.ndarray]
) -> int:
    """Left-compose world-voxel corrections into each tile's index-0 stitching transform.

    The rest of the chain (e.g. the nominal-grid translation) is untouched, so the
    downstream combine step, which copies only transform index 0, receives ``C @ S``
    and each spot channel becomes ``C @ S @ camera @ nominal``.
    """
    tree = ET.parse(source_xml)
    root = tree.getroot()
    mosaic = read_mosaic_xml(source_xml)
    label_by_setup = {tile.setup: label for label, tile in mosaic.tiles.items()}
    updated = 0
    for registration in root.findall(".//ViewRegistrations/ViewRegistration"):
        label = label_by_setup.get(int(registration.get("setup")))
        if label not in corrections:
            continue
        first = registration.find("ViewTransform")
        name = first.findtext("Name") or ""
        if not any(marker in name for marker in STITCHING_TRANSFORM_MARKERS):
            raise ValueError(
                f"Tile {label}: index-0 transform {name!r} is not a stitching transform"
            )
        matrix = np.asarray(corrections[label]) @ parse_affine(first.findtext("affine"))
        first.find("affine").text = format_affine(matrix)
        updated += 1
    if updated != len(corrections):
        raise ValueError(f"Updated {updated} tiles; expected {len(corrections)}")
    tree.write(output_xml, encoding="UTF-8", xml_declaration=True)
    return updated
