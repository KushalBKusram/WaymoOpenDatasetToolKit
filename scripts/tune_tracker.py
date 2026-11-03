"""Tune tracker association settings on held-out validation segments.

Tuning uses validation segments that are not part of the frozen eval set, so the reported
numbers (scripts/track_eval.py on the eval segments) are never used to choose settings.
Selection rule, fixed in advance: highest ground-truth-weighted IDF1, ties broken by MOTA.

  python scripts/tune_tracker.py --config configs/yolov8n.yaml --run-dir runs/yolov8n-v1.2-s42
"""

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from modules.benchmark import load_split_manifest  # noqa: E402
from modules.waymo_open_dataset import ToolKit  # noqa: E402
from track_eval import collect, dev_segments, score  # noqa: E402

GRID = {
    "score_thresh": [0.25, 0.4, 0.5, 0.6],
    "iou_threshold": [0.1, 0.2, 0.3],
    "max_age": [5, 10, 20, 40],
    "min_hits": [1, 2, 3],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--dev-segments", type=int, default=3)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(args.run_dir)
    class_names = cfg["model"].get("class_names", ["vehicle", "pedestrian", "cyclist", "sign"])
    toolkit = ToolKit(split="validation")
    eval_segments = load_split_manifest(cfg["data"]["manifest"])["eval"]
    segments = dev_segments(toolkit, eval_segments, args.dev_segments)
    print("dev segments:", *segments, sep="\n  ", flush=True)

    frames = collect(toolkit, segments, args.camera, cfg, run_dir, args.device)
    rows = []
    for values in itertools.product(*GRID.values()):
        settings = dict(zip(GRID, values))
        report = score(frames, class_names, **settings)
        rows.append({**settings, "mota": report["weighted_mota"], "idf1": report["weighted_idf1"],
                     "vehicle_switches": report["per_class"]["vehicle"]["num_switches"]})
    rows.sort(key=lambda r: (r["idf1"], r["mota"]), reverse=True)
    default = score(frames, class_names)
    result = {"dev_segments": segments, "frames": default["frames"],
              "default": {"settings": default["settings"], "mota": default["weighted_mota"], "idf1": default["weighted_idf1"]},
              "best": rows[0], "top5": rows[:5], "grid": rows}
    out = run_dir / "tracking" / "tuning_dev.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("default on dev:", result["default"])
    for row in rows[:5]:
        print(row)


if __name__ == "__main__":
    main()
