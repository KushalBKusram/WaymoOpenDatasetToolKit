"""Fetch and decode what LiDAR 3-D gating needs for the tuning and evaluation validation segments.

For each segment this reads the LiDAR, LiDAR calibration and vehicle pose components into `.waymo_cache`
(only those, a few hundred MB per segment) and decodes the TOP-lidar frames that are not cached yet, in
parallel across frames (decoding takes about 2 s per frame).

  python scripts/prepare_gate_data.py --manifest configs/benchmark_splits.json --workers 14
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.benchmark import load_split_manifest  # noqa: E402
from modules.waymo_open_dataset import ToolKit  # noqa: E402
from track_eval import dev_segments  # noqa: E402

COMPONENTS = ("lidar", "lidar_calibration", "vehicle_pose", "lidar_box")


def decode_chunk(job: tuple[str, list[int]]) -> int:
    segment, timestamps = job
    toolkit = ToolKit(split="validation")
    toolkit.assign_segment(segment)
    for timestamp in timestamps:
        toolkit.load_lidar_points_xyzi(timestamp, top_only=True)
    return len(timestamps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="configs/benchmark_splits.json")
    parser.add_argument("--dev-segments", type=int, default=3)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    eval_segments = load_split_manifest(args.manifest)["eval"]
    toolkit = ToolKit(split="validation")
    segments = dev_segments(toolkit, eval_segments, args.dev_segments) + list(eval_segments)
    jobs, started = [], time.time()
    for segment in segments:
        toolkit.assign_segment(segment)
        for component in COMPONENTS:
            toolkit._read_cached(component)
        cache = Path(toolkit.cache_dir) / "validation" / "decoded_lidar_top" / segment
        missing = [int(t) for t in sorted(toolkit.get_timestamps()) if not (cache / f"{int(t)}.npy").exists()]
        print(f"{segment[:24]}: components cached, {len(missing)} frames to decode  t={time.time() - started:.0f}s", flush=True)
        jobs += [(segment, missing[i:i + 6]) for i in range(0, len(missing), 6)]
    done = 0
    with Pool(args.workers) as pool:
        for count in pool.imap_unordered(decode_chunk, jobs):
            done += count
            if done % 60 < 6:
                print(f"  decoded {done} frames  t={time.time() - started:.0f}s", flush=True)
    print("GATE_DATA_DONE", flush=True)


if __name__ == "__main__":
    main()
