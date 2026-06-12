from __future__ import annotations
import io
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
import boto3

class DroppedLinksReport:
    def __init__(self, out_prefix: str | Path):
        self.out_prefix = str(out_prefix)

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

    def write(self, rows_sorted, dropped_pairs: Optional[Set[Tuple[int, int]]],
              pair_errors: Dict[Tuple[int, int], float]) -> Optional[str]:

        if not dropped_pairs:
            return None

        corr_by_pair: Dict[Tuple[int, int], list[float]] = {}

        for r in rows_sorted:
            a = int(r[0])
            b = int(r[1])
            corr = float(r[5])
            key = (min(a, b), max(a, b))

            if key in dropped_pairs:
                corr_by_pair.setdefault(key, []).append(corr)

        buf = io.StringIO()
        buf.write("Dropped links summary\n")
        buf.write("=====================\n\n")
        buf.write(f"Total dropped pairs from CSV: {len(dropped_pairs)}\n")
        buf.write(f"Dropped pairs found in XML pairwise results: {len(corr_by_pair)}\n\n")

        missing_pairs = sorted(dropped_pairs - set(corr_by_pair.keys()))

        if missing_pairs:
            buf.write("Dropped pairs NOT found in StitchingResults by TileA/TileB:\n")
            for a, b in missing_pairs:
                err = pair_errors.get((a, b))
                err_str = "N/A" if err is None else f"{err}"
                buf.write(f"  - ({a}, {b})  Error={err_str}\n")
            buf.write("\n")

        if not corr_by_pair:
            buf.write("No dropped pairs had corresponding pairwise correlations in the XML.\n")
        else:
            buf.write("Dropped pairs with error + corr:\n")
            for a, b in sorted(corr_by_pair.keys()):
                best_corr = max(corr_by_pair[(a, b)])
                err = pair_errors.get((a, b))
                err_str = "N/A" if err is None else f"{err}"

                buf.write(f"Pair TileA={a}, TileB={b}:  Error={err_str},  Corr={best_corr}\n")

        text = buf.getvalue()

        if self.out_prefix.startswith("s3://"):
            txt_s3_uri = self.s3_path_join(self.out_prefix, "dropped_links_metrics.txt")
            bucket, key = self.parse_s3_uri(txt_s3_uri)

            s3 = self.create_s3_client()
            s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType="text/plain")

            return txt_s3_uri

        out_dir = Path(self.out_prefix).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        txt_path = out_dir / "dropped_links_metrics.txt"
        txt_path.write_text(text, encoding="utf-8")

        return str(txt_path)