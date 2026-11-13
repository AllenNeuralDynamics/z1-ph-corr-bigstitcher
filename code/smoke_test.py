"""Environment smoke test: imports and a tiny numerical check; never touches /data or /results."""

import importlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> int:
    print(f"[smoke] python {sys.version.split()[0]}")
    for name in ("numpy", "scipy", "pandas", "zarr", "s3fs", "dask", "boto3", "xmltodict",
                 "matplotlib", "aind_data_schema"):
        module = importlib.import_module(name)
        print(f"[smoke] {name} {getattr(module, '__version__', '?')}")

    # Entry modules (imported, not run).
    import run_capsule  # noqa: F401
    from alignment import bigstitcher  # noqa: F401
    from single_channel_xmls import xml_utils  # noqa: F401
    from block_affine import measure, stage  # noqa: F401

    bigstitcher_home = Path(os.environ.get("BIGSTITCHER_HOME", "/home/BigStitcher-Spark"))
    jars = list((bigstitcher_home / "target").glob("*.jar"))
    print(f"[smoke] BigStitcher jars in {bigstitcher_home}/target: {len(jars)}")
    if not jars:
        print("[smoke] FAILED: BigStitcher-Spark is not built")
        return 1

    from scipy.ndimage import shift as image_shift

    rng = np.random.default_rng(0)
    reference = rng.normal(size=(24, 32, 40)).astype(np.float32)
    moving = image_shift(reference, (1.0, -2.0, 3.0), order=1, mode="wrap")
    recovered, snr = measure.phase_correlation_shift(reference, moving, (4, 4, 4))
    if not np.allclose(recovered, (-1.0, 2.0, -3.0), atol=0.2) or snr < 5:
        print(f"[smoke] FAILED: phase correlation returned {recovered}, SNR {snr:.1f}")
        return 1

    import zarr

    with tempfile.TemporaryDirectory() as folder:
        group = zarr.open_group(folder, mode="w")
        group.create_dataset("0", data=np.ones((1, 1, 4, 5, 6), dtype=np.uint16))
        group.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
        view, _ = measure.open_level(folder, 0)
        if view.shape != (4, 5, 6):
            print(f"[smoke] FAILED: OME-Zarr view shape {view.shape}")
            return 1
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
