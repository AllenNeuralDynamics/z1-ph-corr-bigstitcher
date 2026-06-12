from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import boto3
import numpy as np
import xml.etree.ElementTree as ET

class MetricsUtils:
    def is_s3_path(self, path: str | Path) -> bool:
        return str(path).startswith("s3://")

    def create_s3_client(self):
        return boto3.client("s3")

    def parse_s3_uri(self, uri: str | Path) -> Tuple[str, str]:
        uri = str(uri)

        if not uri.startswith("s3://"):
            raise ValueError(f"Expected S3 URI starting with s3://, got {uri!r}")

        without = uri[5:]
        bucket, sep, key = without.partition("/")

        if not bucket or not sep or not key:
            raise ValueError(f"Invalid S3 URI {uri!r}; expected s3://bucket/key")

        return bucket, key

    def load_xml_root(self, xml_path: str | Path) -> ET.Element:
        path_str = str(xml_path)

        if self.is_s3_path(path_str):
            s3 = self.create_s3_client()
            bucket, key = self.parse_s3_uri(path_str)
            resp = s3.get_object(Bucket=bucket, Key=key)
            data = resp["Body"].read()

            return ET.fromstring(data)

        return ET.parse(Path(path_str).expanduser()).getroot()

    def parse_affine_3x4(self, text: str) -> np.ndarray:
        vals = np.array(text.split(), dtype=float)

        if vals.size != 12:
            raise ValueError(f"Expected 12 values in 3x4 affine, got {vals.size}")

        return vals.reshape(3, 4)

    def is_pure_translation(self, aff: np.ndarray, atol: float = 1e-9) -> bool:
        return np.allclose(aff[:, :3], np.eye(3), atol=atol)

    def extract_pairwise_rows(self, root: ET.Element, xy_thresh_log2: float) -> List[list]:
        sr = root.find("StitchingResults")

        if sr is None:
            raise RuntimeError("No <StitchingResults> in XML")

        rows: List[list] = []
        seen = set()

        for pr in sr.findall("PairwiseResult"):
            a = int(pr.get("view_setup_a"))
            b = int(pr.get("view_setup_b"))

            shift_el = pr.find("shift")
            corr_el = pr.find("correlation")
            overlap_el = pr.find("overlap_boundingbox")

            if shift_el is None or corr_el is None or overlap_el is None:
                continue

            if not shift_el.text or not corr_el.text or not overlap_el.text:
                continue

            shift_aff = self.parse_affine_3x4(shift_el.text)

            if not self.is_pure_translation(shift_aff):
                raise RuntimeError(f"Non-translation detected between {a} and {b}")

            shifts = shift_aff[:, 3].astype(float)
            corr = float(corr_el.text)

            bb = np.array(overlap_el.text.split(), dtype=float).reshape(2, 3)

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

            rows.append([a, b, sx_round, sy_round, sz_round, corr_round, float(overlap_x), 
                         float(overlap_y), float(overlap_z), align])

        return rows

    def load_dropped_pairs(self, csv_path: str | Path) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], float]]:
        dropped: Set[Tuple[int, int]] = set()
        pair_errors: Dict[Tuple[int, int], float] = {}

        csv_path_str = str(csv_path)

        try:
            if self.is_s3_path(csv_path_str):
                s3 = self.create_s3_client()
                bucket, key = self.parse_s3_uri(csv_path_str)
                resp = s3.get_object(Bucket=bucket, Key=key)
                text = resp["Body"].read().decode("utf-8")
                f = text.splitlines()
            else:
                local_path = Path(csv_path_str).expanduser()
                with local_path.open("r", newline="") as local_file:
                    text = local_file.read()
                f = text.splitlines()

            sample = "\n".join(f[:100])

            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
                reader = csv.DictReader(f, dialect=dialect)
            except csv.Error:
                reader = csv.DictReader(f)

            fieldnames = reader.fieldnames or []
            has_type = "type" in fieldnames

            for row in reader:
                if not any(row.values()):
                    continue

                if has_type and row.get("type") != "solver_removed_link":
                    continue

                try:
                    a = int(row["a"])
                    b = int(row["b"])
                except (KeyError, ValueError, TypeError):
                    continue

                key = (min(a, b), max(a, b))
                dropped.add(key)

                err_val: Optional[float] = None
                err_str = row.get("error")

                if err_str:
                    try:
                        err_val = float(err_str)
                    except ValueError:
                        err_val = None

                if err_val is not None:
                    if key in pair_errors:
                        pair_errors[key] = max(pair_errors[key], err_val)
                    else:
                        pair_errors[key] = err_val

            print(f"Loaded {len(dropped)} dropped pairs from {csv_path_str}: {sorted(dropped)}")

        except FileNotFoundError:
            print(
                f"[WARN] dropped_links file not found at {csv_path_str}; "
                "continuing without dropped-link labels."
            )

        except Exception as e:
            print(f"[WARN] Error reading dropped_links CSV at {csv_path_str}: {e!r}")

        return dropped, pair_errors