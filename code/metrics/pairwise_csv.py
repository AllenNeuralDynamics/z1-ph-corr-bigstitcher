from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Optional, Set, Tuple
import boto3

class PairwiseCSVWriter:
    def __init__(self, output_path: str | Path):
        self.output_path = str(output_path)

    def parse_s3_uri(self, uri: str) -> Tuple[str, str]:
        if not uri.startswith("s3://"):
            raise ValueError(f"Expected S3 URI starting with s3://, got {uri!r}")

        without = uri[5:]
        bucket, sep, key = without.partition("/")

        if not bucket or not sep or not key:
            raise ValueError(f"Invalid S3 URI {uri!r}; expected s3://bucket/key")

        return bucket, key

    def s3_path_join(self, prefix: str, filename: str) -> str:
        return str(prefix).rstrip("/") + "/" + filename

    def create_s3_client(self):
        return boto3.client("s3")

    def write(
        self,
        rows_sorted,
        dropped_pairs: Optional[Set[Tuple[int, int]]] = None,
    ) -> str:
        base_header = [
            "TileA",
            "TileB",
            "ShiftX",
            "ShiftY",
            "ShiftZ",
            "Correlation",
            "OverlapX",
            "OverlapY",
            "OverlapZ",
            "Alignment",
        ]

        header = (
            base_header + ["DroppedBySolver"]
            if dropped_pairs is not None
            else base_header
        )

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(header)

        for r in rows_sorted:
            if dropped_pairs is None:
                w.writerow(r)
                continue

            a = int(r[0])
            b = int(r[1])
            pair_key = (min(a, b), max(a, b))
            is_dropped = pair_key in dropped_pairs

            w.writerow(list(r) + ["yes" if is_dropped else "no"])

        data = buf.getvalue()

        if self.output_path.startswith("s3://"):
            csv_uri = self.s3_path_join(
                self.output_path,
                "pairwise_links.csv",
            )

            bucket, key = self.parse_s3_uri(csv_uri)

            s3 = self.create_s3_client()
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=data.encode("utf-8"),
                ContentType="text/csv",
            )

            return csv_uri

        out_dir = Path(self.output_path).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "pairwise_links.csv"
        csv_path.write_text(data, encoding="utf-8")

        return str(csv_path)