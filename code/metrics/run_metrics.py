from __future__ import annotations

from pathlib import Path

from .metrics_utils import MetricsUtils
from .pairwise_csv import PairwiseCSVWriter
from .corr_shift_plots import CorrAndShiftPlots
from .dropped_links_report import DroppedLinksReport
from .max_projection import NominalMaxProjectionMosaic

class RunMetrics:
    def __init__(self, dropped_csv_path, xml_path, output_path):
        self.dropped_csv_path = str(dropped_csv_path) if dropped_csv_path else None
        self.xml_path = str(xml_path)
        self.output_path = str(output_path)

        self.xy_thres = 2.0
        self.view_scale_level = "2"
        self.zarr_scale_level = "2"

    def _is_s3_path(self, path: str) -> bool:
        return path.startswith("s3://")

    def _existing_local_or_s3_arg(self, path: str | None) -> str | None:
        if not path:
            return None

        if self._is_s3_path(path):
            return path

        local_path = Path(path).expanduser()
        return str(local_path) if local_path.exists() else None

    def run_alignment_metrics(self) -> None:
        dropped_csv_arg = self._existing_local_or_s3_arg(self.dropped_csv_path)

        self.utils = MetricsUtils()

        root = self.utils.load_xml_root(self.xml_path)
        rows = self.utils.extract_pairwise_rows(root, self.xy_thres)
        rows_sorted = sorted(rows, key=lambda r: r[5], reverse=True)

        dropped_pairs = None
        pair_errors = {}
        dropped_txt_uri = None

        if dropped_csv_arg:
            dropped_pairs_loaded, pair_errors_loaded = self.utils.load_dropped_pairs(dropped_csv_arg)

            if dropped_pairs_loaded:
                dropped_pairs = dropped_pairs_loaded
                pair_errors = pair_errors_loaded
                print(f"✅ Using {len(dropped_pairs)} dropped pair(s) for QC annotations.")

        csv_writer = PairwiseCSVWriter(self.output_path)
        corr_shift = CorrAndShiftPlots(self.output_path)
        max_projection = NominalMaxProjectionMosaic(self.xml_path, self.output_path, self.view_scale_level, self.zarr_scale_level)
        dropped_report = DroppedLinksReport(self.output_path)

        csv_uri = csv_writer.write(rows_sorted, dropped_pairs)

        corr_png_uri, shifts_all_png_uri, shifts_kept_png_uri, split_shift_png_uris = corr_shift.make_plots(
            rows_sorted, dropped_pairs
        )

        max_projection_png_uri, max_projection_links_png_uri = max_projection.run(dropped_pairs)

        if dropped_pairs:
            dropped_txt_uri = dropped_report.write(rows_sorted, dropped_pairs, pair_errors)

        print("QC done.")
        print("  CSV  :", csv_uri)
        print("  Plots:")
        print("    corr vs rank        :", corr_png_uri)
        print("    shifts (all)        :", shifts_all_png_uri)
        print("    shifts (kept)       :", shifts_kept_png_uri)

        for name, uri in split_shift_png_uris.items():
            print(f"    {name:<24}: {uri}")

        print("    max projection      :", max_projection_png_uri)
        print("    max projection links:", max_projection_links_png_uri)

        if dropped_txt_uri:
            print("    dropped link metrics:", dropped_txt_uri)


            