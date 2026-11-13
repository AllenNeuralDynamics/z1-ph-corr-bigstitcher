"""Quality-gated blockwise phase-correlation residuals across tile seams.

Ported from the RhaPSO stitching experiment (Code Ocean capsule 9066422,
``code/phase_correlation.py``); the numerical method is unchanged.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import fft
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.ndimage import shift as image_shift

from .xml_io import MosaicXML


@dataclass(frozen=True)
class SamplingLayout:
    """Where blocks are placed inside each seam overlap."""

    grid_long_z: Tuple[int, int] = (3, 3)
    long_edge_fraction: float = 0.0
    z_edge_fraction: float = 0.25


@dataclass(frozen=True)
class MeasurementSettings:
    pyramid_level: int = 2
    # Level-0 voxels; scaled to the measured pyramid level.
    max_shift_xyz: Tuple[int, int, int] = (12, 12, 6)
    # Level-0 block size along the seam, across the seam (clipped to the overlap) and in Z.
    block_long_short_z: Tuple[int, int, int] = (768, 64, 128)
    min_peak_snr: float = 5.0
    min_correlation_after: float = 0.05
    s3_anonymous: bool = False


class ZarrView:
    """Lazy ZYX view of a TCZYX (or ZYX) OME-Zarr array."""

    def __init__(self, array):
        self._array = array
        self._leading = (0,) * (array.ndim - 3)
        self.shape = tuple(int(value) for value in array.shape[-3:])

    def __getitem__(self, key):
        return self._array[self._leading + tuple(key)]


def open_level(
    uri: str, level: int, anonymous: bool = False
) -> Tuple[ZarrView, np.ndarray]:
    """Open one OME-Zarr pyramid level and return its XYZ level-to-level-0 transform."""
    import zarr

    if uri.startswith("s3://"):
        import s3fs

        filesystem = s3fs.S3FileSystem(anon=anonymous)
        store = s3fs.S3Map(
            root=uri.rstrip("/").replace("s3://", "", 1), s3=filesystem, check=False
        )
    else:
        store = uri
    group = zarr.open_group(store, mode="r")
    datasets = group.attrs["multiscales"][0]["datasets"]
    if level < 0 or level >= len(datasets):
        raise ValueError(f"Pyramid level {level} is unavailable in {uri}")

    def spatial_transform(dataset: dict) -> Tuple[np.ndarray, np.ndarray]:
        scale = np.ones(3)
        translation = np.zeros(3)
        for transform in dataset.get("coordinateTransformations", []):
            if transform["type"] == "scale":
                scale = np.asarray(transform["scale"], dtype=float)[-3:][::-1]
            elif transform["type"] == "translation":
                translation = np.asarray(transform["translation"], dtype=float)[-3:][::-1]
        return scale, translation

    scale_zero, translation_zero = spatial_transform(datasets[0])
    scale_level, translation_level = spatial_transform(datasets[level])
    level_to_zero = np.eye(4)
    level_to_zero[:3, :3] = np.diag(scale_level / scale_zero)
    level_to_zero[:3, 3] = (translation_level - translation_zero) / scale_zero
    return ZarrView(group[datasets[level]["path"]]), level_to_zero


def _transformed_bounds(
    source_shape_zyx: Tuple[int, int, int], source_to_output: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    size_xyz = np.asarray(source_shape_zyx[::-1], dtype=float)
    corners = np.asarray(
        [
            (source_to_output @ np.array([x, y, z, 1.0]))[:3]
            for x in (0.0, size_xyz[0] - 1)
            for y in (0.0, size_xyz[1] - 1)
            for z in (0.0, size_xyz[2] - 1)
        ]
    )
    return corners.min(axis=0), corners.max(axis=0)


def _intersection_bounds(
    shape_a_zyx: Tuple[int, int, int],
    shape_b_zyx: Tuple[int, int, int],
    a_to_b: np.ndarray,
    margin_xyz: Sequence[int] = (12, 8, 12),
) -> Tuple[np.ndarray, np.ndarray]:
    low_a, high_a = _transformed_bounds(shape_a_zyx, a_to_b)
    size_b_xyz = np.asarray(shape_b_zyx[::-1], dtype=float)
    margin = np.asarray(margin_xyz)
    low = np.ceil(np.maximum(low_a, 0)).astype(int) + margin
    high = np.floor(np.minimum(high_a, size_b_xyz - 1)).astype(int) - margin
    if np.any(high <= low):
        raise ValueError(f"No usable XML-seeded overlap: low={low}, high={high}")
    return low, high


def axis_starts(
    low: int, high: int, width: int, count: int, edge_fraction: float = 0
) -> List[int]:
    available = high - low
    if available < width:
        raise ValueError(
            f"Overlap width {available} is smaller than requested block width {width}"
        )
    if count == 1:
        return [low + (available - width) // 2]
    free = available - width
    return (
        np.rint(
            np.linspace(
                low + edge_fraction * free, high - width - edge_fraction * free, count
            )
        )
        .astype(int)
        .tolist()
    )


def _source_bounds(
    output_to_local: np.ndarray,
    low_xyz: np.ndarray,
    high_xyz: np.ndarray,
    shape_zyx: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    corners = np.asarray(
        [
            (output_to_local @ np.array([x, y, z, 1.0]))[:3]
            for x in (low_xyz[0], high_xyz[0] - 1)
            for y in (low_xyz[1], high_xyz[1] - 1)
            for z in (low_xyz[2], high_xyz[2] - 1)
        ]
    )[:, ::-1]
    lower = np.maximum(np.floor(corners.min(axis=0)).astype(int) - 2, 0)
    upper = np.minimum(
        np.ceil(corners.max(axis=0)).astype(int) + 3, np.asarray(shape_zyx)
    )
    return lower, upper


def _sample_volume(
    array: ZarrView,
    output_to_local: np.ndarray,
    low_xyz: np.ndarray,
    high_xyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample one XYZ box into the output frame and return a ZYX volume."""
    x = np.arange(low_xyz[0], high_xyz[0], dtype=np.float32)[None, None, :]
    y = np.arange(low_xyz[1], high_xyz[1], dtype=np.float32)[None, :, None]
    z = np.arange(low_xyz[2], high_xyz[2], dtype=np.float32)[:, None, None]
    shape = (z.shape[0], y.shape[1], x.shape[2])
    local = [
        output_to_local[row, 0] * x
        + output_to_local[row, 1] * y
        + output_to_local[row, 2] * z
        + output_to_local[row, 3]
        for row in range(3)
    ]
    local_x, local_y, local_z = (np.broadcast_to(value, shape) for value in local)
    shape_zyx = tuple(int(value) for value in array.shape)
    lower, upper = _source_bounds(output_to_local, low_xyz, high_xyz, shape_zyx)
    raw = np.asarray(
        array[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]],
        dtype=np.float32,
    )
    coordinates = np.asarray(
        [local_z - lower[0], local_y - lower[1], local_x - lower[2]], dtype=np.float32
    )
    valid = (
        (local_z >= 0)
        & (local_z <= shape_zyx[0] - 1)
        & (local_y >= 0)
        & (local_y <= shape_zyx[1] - 1)
        & (local_x >= 0)
        & (local_x <= shape_zyx[2] - 1)
    )
    sampled = map_coordinates(
        raw, coordinates, order=1, mode="constant", cval=0.0, prefilter=False
    )
    sampled[~valid] = 0
    return sampled, valid


def _preprocess(volume: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = volume[valid]
    low, high = np.quantile(values, [0.005, 0.9995])
    clipped = np.clip(volume, low, high)
    foreground = clipped - gaussian_filter(clipped, sigma=(2, 8, 8))
    foreground[~valid] = 0
    scale = np.std(foreground[valid])
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Phase-correlation block has no intensity variation")
    foreground /= scale
    windows = [np.hanning(length).astype(np.float32) for length in volume.shape]
    window = windows[0][:, None, None] * windows[1][None, :, None]
    window = window * windows[2][None, None, :]
    return np.asarray(foreground * window, dtype=np.float32)


def phase_correlation_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift_zyx: Tuple[int, ...],
    workers: int = 1,
) -> Tuple[np.ndarray, float]:
    """Return the subvoxel ZYX shift to apply to ``moving`` and the peak SNR."""
    reference_fft = fft.fftn(reference, workers=workers)
    moving_fft = fft.fftn(moving, workers=workers)
    cross_power = reference_fft * moving_fft.conj()
    magnitude = np.abs(cross_power)
    cross_power /= np.maximum(magnitude, np.finfo(np.float32).eps)
    correlation = fft.ifftn(cross_power, workers=workers).real

    shifts = [np.arange(-limit, limit + 1, dtype=int) for limit in max_shift_zyx]
    search = correlation[
        np.ix_(*(axis % size for axis, size in zip(shifts, correlation.shape)))
    ]
    peak_index = np.unravel_index(np.argmax(search), search.shape)
    integer_shift = np.asarray(
        [shifts[axis][index] for axis, index in enumerate(peak_index)], dtype=float
    )
    subpixel = integer_shift.copy()
    peak_value = float(search[peak_index])
    for axis in range(reference.ndim):
        index = peak_index[axis]
        if index == 0 or index == search.shape[axis] - 1:
            continue
        minus_index = list(peak_index)
        plus_index = list(peak_index)
        minus_index[axis] -= 1
        plus_index[axis] += 1
        minus = float(search[tuple(minus_index)])
        plus = float(search[tuple(plus_index)])
        denominator = minus - 2 * peak_value + plus
        if denominator != 0:
            offset = 0.5 * (minus - plus) / denominator
            subpixel[axis] += float(np.clip(offset, -0.5, 0.5))

    sidelobes = search.copy()
    slices = tuple(
        slice(max(0, index - 1), min(size, index + 2))
        for index, size in zip(peak_index, search.shape)
    )
    sidelobes[slices] = np.nan
    peak_snr = (peak_value - np.nanmean(sidelobes)) / max(
        np.nanstd(sidelobes), np.finfo(float).eps
    )
    return subpixel, float(peak_snr)


def _overlap_slices(shape: Tuple[int, int, int], shift_zyx: np.ndarray):
    padding = np.ceil(np.abs(shift_zyx)).astype(int) + 2
    return tuple(slice(int(pad), int(size - pad)) for size, pad in zip(shape, padding))


def _correlation(reference: np.ndarray, moving: np.ndarray, valid: np.ndarray) -> float:
    keep = valid & np.isfinite(reference) & np.isfinite(moving)
    if keep.sum() < 2:
        return float("nan")
    return float(np.corrcoef(reference[keep], moving[keep])[0, 1])


OpenLevel = Callable[[str, int], Tuple[ZarrView, np.ndarray]]


def measure_seam(
    mosaic: MosaicXML,
    tile_a: str,
    tile_b: str,
    layout: SamplingLayout,
    settings: MeasurementSettings,
    open_array: OpenLevel,
) -> Tuple[pd.DataFrame, dict]:
    """Residual shifts to apply to XML-placed ``tile_a`` to align it to ``tile_b``.

    Coordinates and shifts are level-0 voxels in ``tile_b``'s local frame.
    """
    info_a, info_b = mosaic.tiles[tile_a], mosaic.tiles[tile_b]
    a_to_b_zero = np.linalg.inv(info_b.matrix) @ info_a.matrix
    array_a, level_to_zero_a = open_array(info_a.image_uri, settings.pyramid_level)
    array_b, level_to_zero_b = open_array(info_b.image_uri, settings.pyramid_level)
    if not np.allclose(level_to_zero_a, level_to_zero_b):
        raise ValueError("Tile pyramid coordinate transforms do not match")
    level_to_zero = level_to_zero_a
    zero_to_level = np.linalg.inv(level_to_zero)
    scale_xyz = np.diag(level_to_zero)[:3]
    a_to_b = zero_to_level @ a_to_b_zero @ level_to_zero
    output_to_a = np.linalg.inv(a_to_b)
    margin_xyz = np.maximum(1, np.ceil(np.asarray([12, 8, 12]) / scale_xyz)).astype(int)
    overlap_low, overlap_high = _intersection_bounds(
        array_a.shape, array_b.shape, a_to_b, margin_xyz=margin_xyz
    )
    available_xyz = overlap_high - overlap_low
    long_axis = int(np.argmax(available_xyz[:2]))
    short_axis = 1 - long_axis
    block_zero_xyz = np.zeros(3)
    block_zero_xyz[long_axis], block_zero_xyz[short_axis], block_zero_xyz[2] = (
        settings.block_long_short_z
    )
    block_xyz = np.maximum(1, np.rint(block_zero_xyz / scale_xyz).astype(int))
    block_xyz[short_axis] = min(int(block_xyz[short_axis]), int(available_xyz[short_axis]))
    if np.any(block_xyz > available_xyz):
        raise ValueError(f"Block {block_xyz.tolist()} exceeds overlap {available_xyz.tolist()}")
    long_starts = axis_starts(
        overlap_low[long_axis],
        overlap_high[long_axis],
        int(block_xyz[long_axis]),
        layout.grid_long_z[0],
        edge_fraction=layout.long_edge_fraction,
    )
    short_start = axis_starts(
        overlap_low[short_axis], overlap_high[short_axis], int(block_xyz[short_axis]), 1
    )[0]
    z_starts = axis_starts(
        overlap_low[2],
        overlap_high[2],
        int(block_xyz[2]),
        layout.grid_long_z[1],
        edge_fraction=layout.z_edge_fraction,
    )
    max_shift_level_xyz = np.maximum(
        1, np.ceil(np.asarray(settings.max_shift_xyz) / scale_xyz).astype(int)
    )
    max_shift_zyx = tuple(int(value) for value in reversed(max_shift_level_xyz))
    axis_name = "X" if long_axis == 0 else "Y"
    rows = []
    for z_index, z_start in enumerate(z_starts):
        for long_index, long_start in enumerate(long_starts):
            low_xyz = np.empty(3, dtype=int)
            low_xyz[long_axis] = long_start
            low_xyz[short_axis] = short_start
            low_xyz[2] = z_start
            high_xyz = low_xyz + block_xyz
            moving, valid_a = _sample_volume(array_a, output_to_a, low_xyz, high_xyz)
            reference, valid_b = _sample_volume(array_b, np.eye(4), low_xyz, high_xyz)
            valid = valid_a & valid_b
            row = {
                "z_index": z_index,
                "long_index": long_index,
                "long_axis": axis_name,
                "block_label": f"Z{z_index + 1}-{axis_name}{long_index + 1}",
                "valid_fraction": float(valid.mean()),
            }
            low_zero_xyz = (level_to_zero @ np.r_[low_xyz.astype(float), 1.0])[:3]
            row.update(
                {"x_start": low_zero_xyz[0], "y_start": low_zero_xyz[1], "z_start": low_zero_xyz[2]}
            )
            try:
                processed_moving = _preprocess(moving, valid)
                processed_reference = _preprocess(reference, valid)
            except ValueError as error:
                row.update({"accepted": False, "rejection": str(error)})
                rows.append(row)
                continue
            shift_zyx, peak_snr = phase_correlation_shift(
                processed_reference, processed_moving, max_shift_zyx=max_shift_zyx
            )
            shifted = image_shift(
                moving, shift_zyx, order=1, mode="constant", cval=0, prefilter=False
            )
            shifted_processed = image_shift(
                processed_moving, shift_zyx, order=1, mode="constant", cval=0, prefilter=False
            )
            crop = _overlap_slices(moving.shape, shift_zyx)
            correlation_before = _correlation(reference[crop], moving[crop], valid[crop])
            correlation_after = _correlation(reference[crop], shifted[crop], valid[crop])
            feature_before = _correlation(
                processed_reference[crop], processed_moving[crop], valid[crop]
            )
            feature_after = _correlation(
                processed_reference[crop], shifted_processed[crop], valid[crop]
            )
            shift_level_xyz = shift_zyx[::-1]
            shift_xyz = level_to_zero[:3, :3] @ shift_level_xyz
            shift_um_xyz = shift_xyz * np.asarray(mosaic.voxel_size_xyz)
            at_boundary = any(
                abs(value) >= limit - 0.51
                for value, limit in zip(shift_level_xyz, max_shift_level_xyz)
            )
            reasons = []
            if at_boundary:
                reasons.append("peak at search boundary")
            if peak_snr < settings.min_peak_snr:
                reasons.append("low peak SNR")
            if not correlation_after >= settings.min_correlation_after:
                reasons.append("low correlation")
            if not correlation_after > correlation_before:
                reasons.append("correlation not improved")
            if not feature_after > feature_before:
                reasons.append("high-pass correlation not improved")
            row.update(
                {
                    "dx_vox": shift_xyz[0],
                    "dy_vox": shift_xyz[1],
                    "dz_vox": shift_xyz[2],
                    "dx_um": shift_um_xyz[0],
                    "dy_um": shift_um_xyz[1],
                    "dz_um": shift_um_xyz[2],
                    "magnitude_um": float(np.linalg.norm(shift_um_xyz)),
                    "phase_peak_snr": peak_snr,
                    "correlation_before": correlation_before,
                    "correlation_after": correlation_after,
                    "feature_correlation_before": feature_before,
                    "feature_correlation_after": feature_after,
                    "accepted": not reasons,
                    "rejection": "; ".join(reasons),
                }
            )
            rows.append(row)
    table = pd.DataFrame(rows)
    summary = {
        "seam_axis": "Y" if long_axis == 0 else "X",
        "level": settings.pyramid_level,
        "downsampling_xyz": scale_xyz.tolist(),
        "block_xyz_vox": (block_xyz * scale_xyz).tolist(),
        "blocks": len(table),
        "accepted_blocks": int(table["accepted"].sum()),
    }
    return table, summary


def measure_mosaic(
    mosaic: MosaicXML,
    seams: Sequence[Tuple[str, str]],
    layout: SamplingLayout,
    settings: MeasurementSettings,
    workers: int = 1,
    open_array: Optional[OpenLevel] = None,
    log: Callable[[str], None] = print,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Measure every seam in parallel; failed seams are reported, not raised."""
    cache: Dict[Tuple[str, int], Tuple[ZarrView, np.ndarray]] = {}
    lock = threading.Lock()

    def cached_open(uri: str, level: int):
        key = (uri, level)
        with lock:
            if key not in cache:
                if open_array is not None:
                    cache[key] = open_array(uri, level)
                else:
                    cache[key] = open_level(uri, level, settings.s3_anonymous)
            return cache[key]

    def run(seam: Tuple[str, str]):
        tile_a, tile_b = seam
        name = f"{tile_a}_{tile_b}"
        started = time.time()
        try:
            table, summary = measure_seam(
                mosaic, tile_a, tile_b, layout, settings, cached_open
            )
        except Exception as error:  # noqa: BLE001 - a failed seam becomes QC, not a crash
            log(f"[block-affine] seam {name} failed: {type(error).__name__}: {error}")
            return None, {
                "seam": name, "tile_a": tile_a, "tile_b": tile_b,
                "blocks": 0, "accepted_blocks": 0, "error": f"{type(error).__name__}: {error}",
            }
        table.insert(0, "seam", name)
        table.insert(1, "tile_a", tile_a)
        table.insert(2, "tile_b", tile_b)
        for axis, value in zip("xyz", summary["block_xyz_vox"]):
            table[f"block_{axis}_vox"] = value
        log(
            f"[block-affine] seam {name}: {summary['accepted_blocks']}/{summary['blocks']} "
            f"accepted ({time.time() - started:.1f} s)"
        )
        return table, {"seam": name, "tile_a": tile_a, "tile_b": tile_b, **summary, "error": ""}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        results = list(executor.map(run, seams))
    tables = [table for table, _ in results if table is not None]
    blocks = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    seam_table = pd.DataFrame([summary for _, summary in results])
    return blocks, seam_table
