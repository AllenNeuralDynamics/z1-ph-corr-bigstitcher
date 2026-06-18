from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import boto3
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import s3fs
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


class NominalMaxProjectionMosaic:
    def __init__(self, xml_path: str | Path, out_dir: str | Path, view_scale_level: str, zarr_scale_level: str):
        self.xml_path = str(xml_path)
        self.out_dir = str(out_dir).rstrip("/")
        self.view_scale_level = str(view_scale_level)
        self.zarr_scale_level = str(zarr_scale_level)

        self.view_scale_float = self._safe_scale_float(self.view_scale_level, default=2.0)
        self.zarr_scale_float = self._safe_scale_float(self.zarr_scale_level, default=4.0)
        self.annotation_scale = self.view_scale_float / self.zarr_scale_float

        self.s3 = boto3.client("s3") if self.is_s3_path(self.out_dir) or self.is_s3_path(self.xml_path) else None

        self.depth_cmap = LinearSegmentedColormap.from_list(
            "depth_reference_orange_magenta_plum",
            [
                (0.00, "#f0c391"),
                (0.20, "#c99a68"),
                (0.40, "#b86135"),
                (0.60, "#b73682"),
                (0.80, "#8f4c7f"),
                (1.00, "#d7b6d8"),
            ],
        )

        if not self.is_s3_path(self.out_dir):
            Path(self.out_dir).expanduser().mkdir(parents=True, exist_ok=True)

    def _safe_scale_float(self, value: str, default: float) -> float:
        try:
            parsed = float(value)
            return parsed if parsed > 0 else default
        except Exception:
            return default

    def is_s3_path(self, path: str | Path) -> bool:
        return str(path).startswith("s3://")

    def parse_s3_uri(self, uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        return parsed.netloc, parsed.path.lstrip("/")

    def output_uri(self, filename: str) -> str:
        if self.is_s3_path(self.out_dir):
            return f"{self.out_dir}/{filename}"
        return str(Path(self.out_dir).expanduser() / filename)

    def save_array_png(self, arr: np.ndarray, filename: str, cmap="gray", vmin=0, vmax=1) -> str:
        uri = self.output_uri(filename)

        if self.is_s3_path(uri):
            bucket, key = self.parse_s3_uri(uri)
            buf = io.BytesIO()

            if arr.ndim == 3:
                plt.imsave(buf, np.clip(arr, 0, 1), format="png")
            else:
                plt.imsave(buf, arr, cmap=cmap, vmin=vmin, vmax=vmax, format="png")

            buf.seek(0)
            self.s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue(), ContentType="image/png")
            return uri

        if arr.ndim == 3:
            plt.imsave(uri, np.clip(arr, 0, 1))
        else:
            plt.imsave(uri, arr, cmap=cmap, vmin=vmin, vmax=vmax)

        return uri

    def save_figure_png(self, fig, filename: str, dpi=250, bbox_inches="tight", pad_inches=0.12) -> str:
        uri = self.output_uri(filename)

        if self.is_s3_path(uri):
            bucket, key = self.parse_s3_uri(uri)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
            buf.seek(0)
            self.s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue(), ContentType="image/png")
            return uri

        fig.savefig(uri, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
        return uri

    def load_xml_root(self, xml_path: str) -> ET.Element:
        xml_path = str(xml_path)

        if self.is_s3_path(xml_path):
            bucket, key = self.parse_s3_uri(xml_path)
            obj = self.s3.get_object(Bucket=bucket, Key=key)
            return ET.fromstring(obj["Body"].read())

        return ET.parse(xml_path).getroot()

    def parse_tile_indices(self, tile_name: str):
        m = re.search(r"Tile_X_(\d+)_Y_(\d+)_Z_(\d+)", tile_name)

        if not m:
            raise RuntimeError(f"Could not parse tile indices from: {tile_name}")

        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    def affine_12_to_4x4(self, affine_text: str) -> np.ndarray:
        vals = [float(v) for v in affine_text.split()]

        if len(vals) != 12:
            raise RuntimeError(f"Expected 12 affine values, got {len(vals)}: {affine_text}")

        mat = np.eye(4, dtype=np.float64)
        mat[0, 0:4] = vals[0:4]
        mat[1, 0:4] = vals[4:8]
        mat[2, 0:4] = vals[8:12]
        return mat

    def parse_affine_3x4(self, text: str) -> np.ndarray:
        vals = np.array(text.split(), dtype=float)

        if vals.size != 12:
            raise ValueError(f"Expected 12 values in 3x4 affine, got {vals.size}")

        return vals.reshape(3, 4)

    def is_pure_translation(self, aff: np.ndarray, atol: float = 1e-9) -> bool:
        return np.allclose(aff[:, :3], np.eye(3), atol=atol)

    def parse_view_setup_sizes(self, root: ET.Element):
        sizes = {}

        for vs in root.findall(".//ViewSetup"):
            setup_id_text = vs.findtext("id")
            size_text = vs.findtext("size")

            if setup_id_text is None or size_text is None:
                continue

            setup_id = int(setup_id_text)
            sx, sy, sz = [int(v) for v in size_text.split()]
            sizes[setup_id] = {"size_x": sx, "size_y": sy, "size_z": sz}

        return sizes

    def parse_named_transforms_from_xml(self, root: ET.Element):
        transform_name = "Translation to Nominal Grid"
        transforms = {}

        for vr in root.findall(".//ViewRegistration"):
            setup = int(vr.get("setup"))
            tp = int(vr.get("timepoint", 0))

            if tp != 0:
                continue

            for vt in vr.findall("ViewTransform"):
                name = vt.findtext("Name")
                affine_text = vt.findtext("affine")

                if name == transform_name and affine_text:
                    transforms[setup] = self.affine_12_to_4x4(affine_text)
                    break

        return transforms

    def parse_zarr_tile_paths_from_xml(self, root: ET.Element):
        image_loader = root.find(".//ImageLoader")

        if image_loader is None:
            raise RuntimeError("No <ImageLoader> found in XML")

        zarr_base = image_loader.findtext("zarr")

        if not zarr_base:
            raise RuntimeError("No <zarr> base path found in XML ImageLoader")

        zarr_base = zarr_base.rstrip("/") + "/"
        tile_records = []

        for zg in image_loader.findall(".//zgroup"):
            rel_path = zg.get("path")

            if not rel_path:
                continue

            x_idx, y_idx, z_idx = self.parse_tile_indices(rel_path)

            tile_records.append({
                "setup": int(zg.get("setup")),
                "tp": int(zg.get("tp", 0)),
                "path": rel_path,
                "full_path": zarr_base + rel_path,
                "tile_x": x_idx,
                "tile_y": y_idx,
                "tile_z": z_idx,
            })

        return tile_records

    def extract_pairwise_rows(self, root: ET.Element, xy_thresh_log2: float):
        sr = root.find(".//StitchingResults")

        if sr is None:
            return []

        rows = []
        seen = set()

        for pr in sr.findall("PairwiseResult"):
            a = int(pr.get("view_setup_a"))
            b = int(pr.get("view_setup_b"))

            shift_text = pr.findtext("shift")
            corr_text = pr.findtext("correlation")
            overlap_text = pr.findtext("overlap_boundingbox")

            if not shift_text or not corr_text or not overlap_text:
                continue

            shift_aff = self.parse_affine_3x4(shift_text)

            if not self.is_pure_translation(shift_aff):
                raise RuntimeError(f"Non-translation detected between {a} and {b}")

            shifts = shift_aff[:, 3].astype(float)
            corr = float(corr_text)

            bb = np.array(overlap_text.split(), dtype=float).reshape(2, 3)
            ext = bb[1] - bb[0]
            overlap_x, overlap_y, overlap_z = ext.tolist()

            eps = 1e-9
            x = max(abs(overlap_x), eps)
            y = max(abs(overlap_y), eps)

            if np.log2(x / y) > xy_thresh_log2:
                align = "top_bottom"
            elif np.log2(y / x) > xy_thresh_log2:
                align = "left_right"
            else:
                align = "corner"

            sx_round = round(float(shifts[0]), 3)
            sy_round = round(float(shifts[1]), 3)
            sz_round = round(float(shifts[2]), 3)
            corr_round = round(corr, 6)

            key = (min(a, b), max(a, b), sx_round, sy_round, sz_round, corr_round)

            if key in seen:
                continue

            seen.add(key)

            rows.append([
                a, b, sx_round, sy_round, sz_round, corr_round,
                float(overlap_x), float(overlap_y), float(overlap_z), align,
            ])

        return rows

    def open_ome_zarr_level(self, zarr_path: str):
        if self.is_s3_path(zarr_path):
            s3_fs = s3fs.S3FileSystem(anon=False)
            store = s3fs.S3Map(root=zarr_path.rstrip("/"), s3=s3_fs, check=False)
            return da.from_zarr(store, component=self.zarr_scale_level)

        return da.from_zarr(zarr_path, component=self.zarr_scale_level)

    def max_projection_and_depth_index(self, vol_zyx: da.Array | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_yx = vol_zyx.max(axis=0).compute().astype(np.float32)
        z_idx_yx = vol_zyx.argmax(axis=0).compute().astype(np.float32)
        return max_yx, z_idx_yx

    def estimate_global_contrast_limits(self, tile_max_projections: list[np.ndarray]) -> tuple[float, float]:
        sample_values = []

        for proj in tile_max_projections:
            arr = proj.astype(np.float32, copy=False)
            mask = np.isfinite(arr) & (arr > 0)

            if not np.any(mask):
                continue

            values = arr[mask]

            if values.size > 250_000:
                step = max(1, values.size // 250_000)
                values = values[::step]

            sample_values.append(values)

        if not sample_values:
            return 0.0, 1.0

        pooled = np.concatenate(sample_values)

        lo = float(np.percentile(pooled, 20))
        hi = float(np.percentile(pooled, 99.5))

        if hi <= lo:
            lo = float(np.nanmin(pooled))
            hi = float(np.nanmax(pooled))

        if hi <= lo:
            hi = lo + 1.0

        return lo, hi

    def contrast_stretch_with_limits(self, arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
        arr = arr.astype(np.float32, copy=False)
        finite = np.isfinite(arr)

        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)

        out = (arr - lo) / (hi - lo)
        out = np.clip(out, 0, 1)
        out[~finite] = 0
        return out.astype(np.float32)

    def depth_encoded_projection_from_raw(self, max_yx: np.ndarray, z_idx_yx: np.ndarray, z_count: int, lo: float, hi: float):
        intensity_yx = self.contrast_stretch_with_limits(max_yx, lo, hi)

        if z_count <= 1:
            depth_yx = np.zeros_like(z_idx_yx, dtype=np.float32)
        else:
            depth_yx = z_idx_yx.astype(np.float32) / float(z_count - 1)

        rgb = self.depth_cmap(depth_yx)[..., :3].astype(np.float32)

        brightness = np.clip(intensity_yx, 0, 1) ** 0.82
        rgb *= brightness[..., None] * 0.82

        empty = max_yx <= 0
        rgb[empty, :] = 0

        return rgb

    def edge_color(self, corr: float) -> str:
        if corr >= 0.90:
            return "#00b7ff"
        if corr >= 0.80:
            return "#00ff66"
        if corr >= 0.70:
            return "#ffd400"
        return "#ff2d2d"

    def add_depth_colorbar(self, fig, ax, z_count: int | None = None, pad: float = 0.025):
        sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=self.depth_cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.032, pad=pad, shrink=0.38)
        cbar.outline.set_edgecolor("white")
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(colors="white", labelsize=8, length=3)

        if z_count is not None and z_count > 1:
            cbar.set_label(f"Depth in Z plane, 0 → {z_count - 1}", color="white", fontsize=9, labelpad=4)
            cbar.set_ticks([0, 0.5, 1.0])
            cbar.set_ticklabels(["z=0", f"z={int((z_count - 1) / 2)}", f"z={z_count - 1}"])
        else:
            cbar.set_label("Depth in Z", color="white", fontsize=9, labelpad=4)
            cbar.set_ticks([0, 0.5, 1.0])
            cbar.set_ticklabels(["shallow", "mid", "deep"])

        for text in cbar.ax.get_xticklabels():
            text.set_color("white")

    def save_depth_mosaic_figure(self, mosaic_rgb: np.ndarray, output_name: str, z_count: int | None = None):
        if mosaic_rgb is None:
            return None

        mosaic_h, mosaic_w = mosaic_rgb.shape[:2]

        fig_w = max(8.0, min(24.0, mosaic_w / 350.0))
        fig_h = max(8.0, min(24.0, mosaic_h / 350.0))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="black")
        ax.set_facecolor("black")
        ax.imshow(np.clip(mosaic_rgb, 0, 1), origin="upper")

        ax.set_title("Depth-colored max projection", color="white", pad=14)
        ax.set_axis_off()

        self.add_depth_colorbar(fig, ax, z_count=z_count, pad=0.030)
        fig.tight_layout(rect=[0.0, 0.035, 1.0, 0.970])
        uri = self.save_figure_png(fig, output_name, dpi=250, bbox_inches="tight", pad_inches=0.12)
        
        plt.close(fig)
        return uri

    def build_nominal_mosaic(self, tile_proj_records, output_name: str, z_count: int | None = None):
        if not tile_proj_records:
            return None, {}, {}, None

        placed = []

        for rec in tile_proj_records:
            proj = rec["projection_raw"]
            tile_h, tile_w = proj.shape

            scale_x = rec["full_size_x"] / tile_w
            scale_y = rec["full_size_y"] / tile_h

            nominal = rec["nominal_transform"]
            tx_full = float(nominal[0, 3])
            ty_full = float(nominal[1, 3])

            x0 = int(round(tx_full / scale_x))
            y0 = int(round(ty_full / scale_y))

            placed.append({
                **rec,
                "x0": x0,
                "y0": y0,
                "x1": x0 + tile_w,
                "y1": y0 + tile_h,
                "tile_w": tile_w,
                "tile_h": tile_h,
            })

        min_x = min(r["x0"] for r in placed)
        min_y = min(r["y0"] for r in placed)
        max_x = max(r["x1"] for r in placed)
        max_y = max(r["y1"] for r in placed)

        mosaic_raw = np.zeros((max_y - min_y, max_x - min_x), dtype=np.float32)
        mosaic_rgb = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=np.float32)
        centers = {}
        tile_bounds = {}

        for rec in placed:
            proj = rec["projection_raw"]
            rgb = rec["projection_rgb"]

            y0 = rec["y0"] - min_y
            y1 = rec["y1"] - min_y
            x0 = rec["x0"] - min_x
            x1 = rec["x1"] - min_x

            region_raw = mosaic_raw[y0:y1, x0:x1]
            region_rgb = mosaic_rgb[y0:y1, x0:x1, :]

            take = proj > region_raw
            region_raw[take] = proj[take]
            region_rgb[take, :] = rgb[take, :]

            centers[rec["setup"]] = {"cx": x0 + 0.5 * rec["tile_w"], "cy": y0 + 0.5 * rec["tile_h"]}
            tile_bounds[rec["setup"]] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

        nonzero = mosaic_raw > 0
        mosaic_rgb[~nonzero, :] = 0
        mosaic_rgb = np.clip(mosaic_rgb * 1.08, 0, 1)

        mosaic_uri = self.save_depth_mosaic_figure(mosaic_rgb, output_name, z_count=z_count)
        return mosaic_rgb, centers, tile_bounds, mosaic_uri

    def draw_pairwise_links_on_mosaic(self, mosaic_png: np.ndarray, centers, rows, output_name: str, tile_bounds=None, z_count: int | None = None):
        if mosaic_png is None or not centers:
            return None

        best_row_by_pair = {}

        for r in rows:
            a = int(r[0])
            b = int(r[1])
            corr = float(r[5])
            key = (min(a, b), max(a, b))

            if key not in best_row_by_pair or corr > float(best_row_by_pair[key][5]):
                best_row_by_pair[key] = r

        mosaic_h, mosaic_w = mosaic_png.shape[:2]

        fig_w = max(8.0, min(24.0, mosaic_w / 350.0))
        fig_h = max(8.0, min(24.0, mosaic_h / 350.0))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="black")
        ax.set_facecolor("black")
        ax.imshow(np.clip(mosaic_png, 0, 1), origin="upper")

        style = max(0.25, min(1.0, self.annotation_scale))
        line_width = max(0.35, 1.15 * style)
        bounds_width = max(0.25, 0.7 * style)
        marker_line_width = max(0.25, 0.9 * style)
        marker_area_scale = max(0.25, style * style)
        font_scale = max(0.60, style)

        if tile_bounds:
            for bounds in tile_bounds.values():
                x0 = bounds["x0"]
                y0 = bounds["y0"]
                width = bounds["x1"] - bounds["x0"]
                height = bounds["y1"] - bounds["y0"]

                rect = plt.Rectangle(
                    (x0, y0), width, height, fill=False,
                    edgecolor="white", linewidth=bounds_width, alpha=0.35,
                    linestyle="--", zorder=3,
                )
                ax.add_patch(rect)

        for r in best_row_by_pair.values():
            a = int(r[0])
            b = int(r[1])
            corr = float(r[5])

            if a not in centers or b not in centers:
                continue

            ax.plot(
                [centers[a]["cx"], centers[b]["cx"]],
                [centers[a]["cy"], centers[b]["cy"]],
                color=self.edge_color(corr),
                linewidth=line_width,
                alpha=0.90,
                solid_capstyle="round",
                zorder=4,
            )

        n_tiles = len(centers)

        if n_tiles <= 30:
            marker_size = 150
            font_size = 8
        elif n_tiles <= 80:
            marker_size = 100
            font_size = 6
        else:
            marker_size = 65
            font_size = 5

        marker_size *= marker_area_scale
        font_size *= font_scale

        for setup, c in centers.items():
            ax.scatter(
                c["cx"], c["cy"], s=marker_size, color="white", edgecolors="black",
                marker="s", linewidths=marker_line_width, alpha=0.90, zorder=5,
            )

            ax.text(
                c["cx"], c["cy"], str(setup), fontsize=font_size, color="black",
                ha="center", va="center", alpha=0.95, zorder=6,
            )

        legend_handles = [
            Line2D([0], [0], color="#00b7ff", lw=max(1.0, 2.0 * style), alpha=0.95, label="corr ≥ 0.90"),
            Line2D([0], [0], color="#00ff66", lw=max(1.0, 2.0 * style), alpha=0.95, label="0.80 ≤ corr < 0.90"),
            Line2D([0], [0], color="#ffd400", lw=max(1.0, 2.0 * style), alpha=0.95, label="0.70 ≤ corr < 0.80"),
            Line2D([0], [0], color="#ff2d2d", lw=max(1.0, 2.0 * style), alpha=0.95, label="corr < 0.70"),
        ]

        if tile_bounds:
            legend_handles.append(
                Line2D([0], [0], color="white", lw=max(0.7, 1.0 * style), alpha=0.55, linestyle="--", label="tile bounds")
            )
        
        legend_ncol = 5 if tile_bounds else 4

        # Put title and legend in separate figure-level bands.
        # Using fig.suptitle avoids fighting with ax.set_title + tight_layout.
        fig.suptitle(
            "Depth-colored max projection with pairwise links",
            color="white",
            fontsize=12,
            y=0.985,
        )

        legend = fig.legend(
            handles=legend_handles,
            title="Link bands",
            loc="upper center",
            ncol=legend_ncol,
            bbox_to_anchor=(0.5, 0.945),
            frameon=True,
        )

        legend.get_frame().set_facecolor("black")
        legend.get_frame().set_edgecolor("white")
        legend.get_frame().set_alpha(0.75)
        legend.get_title().set_color("white")

        for text in legend.get_texts():
            text.set_color("white")

        ax.set_axis_off()

        self.add_depth_colorbar(fig, ax, z_count=z_count, pad=0.030)

        fig.tight_layout(rect=[0.0, 0.055, 1.0, 0.875])

        uri = self.save_figure_png(fig, output_name, dpi=250, bbox_inches="tight", pad_inches=0.12)

        plt.close(fig)
        return uri

    def run(self):
        root = self.load_xml_root(self.xml_path)

        setup_sizes = self.parse_view_setup_sizes(root)
        nominal_transforms = self.parse_named_transforms_from_xml(root)
        tile_records = self.parse_zarr_tile_paths_from_xml(root)
        tile_records = sorted(tile_records, key=lambda r: r["setup"])

        missing_transforms = [r["setup"] for r in tile_records if r["setup"] not in nominal_transforms]
        if missing_transforms:
            raise RuntimeError(
                f"Missing transform 'Translation to Nominal Grid' for setups: "
                f"{missing_transforms[:20]}{'...' if len(missing_transforms) > 20 else ''}"
            )

        missing_sizes = [r["setup"] for r in tile_records if r["setup"] not in setup_sizes]
        if missing_sizes:
            raise RuntimeError(
                f"Missing ViewSetup size for setups: "
                f"{missing_sizes[:20]}{'...' if len(missing_sizes) > 20 else ''}"
            )

        print(f"Using zarr_scale_level={self.zarr_scale_level} for image data.")
        print(f"Using view_scale_level={self.view_scale_level} for annotation styling.")
        print(f"Annotation scale factor: {self.annotation_scale:.3f}")

        tile_proj_records = []
        tile_max_projections = []
        z_count_for_key = None

        for rec in tile_records:
            setup = rec["setup"]
            tile_path = rec["full_path"]
            nominal = nominal_transforms[setup]
            size_info = setup_sizes[setup]

            print(f"Building raw max projection for setup {setup}: {tile_path}")

            arr = self.open_ome_zarr_level(tile_path)

            if arr.ndim == 5:
                vol_zyx = arr[0, 0, :, :, :].astype(np.float32)
            elif arr.ndim == 3:
                vol_zyx = arr.astype(np.float32)
            else:
                raise RuntimeError(
                    f"Unexpected array shape for {tile_path}: {arr.shape}. "
                    "Expected 5D T,C,Z,Y,X or 3D Z,Y,X."
                )

            z_count = int(vol_zyx.shape[0])

            if z_count_for_key is None:
                z_count_for_key = z_count

            proj_yx, z_idx_yx = self.max_projection_and_depth_index(vol_zyx)
            tile_max_projections.append(proj_yx)

            tile_proj_records.append({
                "setup": setup,
                "tile_x": rec["tile_x"],
                "tile_y": rec["tile_y"],
                "tile_z": rec["tile_z"],
                "projection_raw": proj_yx,
                "projection_z_idx": z_idx_yx,
                "projection_rgb": None,
                "nominal_transform": nominal,
                "full_size_x": size_info["size_x"],
                "full_size_y": size_info["size_y"],
                "full_size_z": size_info["size_z"],
                "z_count": z_count,
            })

        global_lo, global_hi = self.estimate_global_contrast_limits(tile_max_projections)
        print(f"Using global max-projection contrast limits: lo={global_lo:.3f}, hi={global_hi:.3f}")

        for rec in tile_proj_records:
            rec["projection_rgb"] = self.depth_encoded_projection_from_raw(
                max_yx=rec["projection_raw"],
                z_idx_yx=rec["projection_z_idx"],
                z_count=rec["z_count"],
                lo=global_lo,
                hi=global_hi,
            )

        mosaic_png, centers, tile_bounds, max_projection_uri = self.build_nominal_mosaic(
            tile_proj_records=tile_proj_records,
            output_name="max_projection.png",
            z_count=z_count_for_key,
        )

        rows = self.extract_pairwise_rows(root, xy_thresh_log2=2.0)

        max_projection_links_uri = self.draw_pairwise_links_on_mosaic(
            mosaic_png=mosaic_png,
            centers=centers,
            rows=rows,
            output_name="max_projection_with_links.png",
            tile_bounds=tile_bounds,
            z_count=z_count_for_key,
        )

        return max_projection_uri, max_projection_links_uri