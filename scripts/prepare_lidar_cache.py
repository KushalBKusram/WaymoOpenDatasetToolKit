"""Pre-decode TOP-lidar frames for the benchmark segments into .waymo_cache.

Decoding one LiDAR frame from its range images takes about 2 s, so training
and evaluation read cached copies. Run once per machine, in parallel:

  python scripts/prepare_lidar_cache.py --workers 16
"""

import argparse
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.benchmark import load_split_manifest  # noqa: E402
from modules.waymo_open_dataset import ToolKit  # noqa: E402


def decode_segment(job: tuple[str, str]) -> tuple[str, str, int]:
    split, segment = job
    toolkit = ToolKit(split=split)
    toolkit.assign_segment(segment)
    timestamps = toolkit.get_timestamps()
    for timestamp in timestamps:
        toolkit.load_lidar_points_xyzi(timestamp, top_only=True)
    return split, segment, len(timestamps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="configs/benchmark_splits.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    manifest = load_split_manifest(args.manifest)
    jobs = [("training", s) for s in manifest["train"]] + [("validation", s) for s in manifest["eval"]]
    start = time.time()
    with Pool(args.workers) as pool:
        for done, (split, segment, frames) in enumerate(pool.imap_unordered(decode_segment, jobs), 1):
            print(f"[{done}/{len(jobs)}] {split} {segment[:24]} {frames} frames  t={time.time() - start:.0f}s",
                  flush=True)
    print("LIDAR_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
