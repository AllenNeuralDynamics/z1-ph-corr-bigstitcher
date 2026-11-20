"""Run the block-affine refinement on an existing BigStitcher XML.

Example (public data, no credentials):
    python -m block_affine bigstitcher.xml out/ --pyramid-level 2 --anonymous
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .stage import BlockAffineConfig, _json_default, refine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seed_xml", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pyramid-level", type=int, default=2)
    parser.add_argument("--holdout-pyramid-level", type=int)
    parser.add_argument("--anchor-tile")
    parser.add_argument("--fixed-ridge", type=float)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CO_CPUS", "4")))
    parser.add_argument("--anonymous", action="store_true", help="unsigned S3 reads")
    args = parser.parse_args()
    config = BlockAffineConfig(
        pyramid_level=args.pyramid_level,
        holdout_pyramid_level=args.holdout_pyramid_level,
        anchor_tile=args.anchor_tile,
        fixed_ridge=args.fixed_ridge,
        s3_anonymous=args.anonymous,
    )
    result = refine(args.seed_xml, args.output_dir, config, workers=args.workers)
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
