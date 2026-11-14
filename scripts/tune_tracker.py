"""Tune tracker association settings on held-out validation segments.

Tuning uses validation segments that are not part of the frozen eval set, so the reported
numbers (scripts/track_eval.py on the eval segments) are never used to choose settings.
Selection rule, fixed in advance: highest ground-truth-weighted IDF1, ties broken by MOTA.

  python scripts/tune_tracker.py --config configs/yolov8n.yaml --run-dir runs/yolov8n-v1.2-s42

With `--reid` the grid also includes appearance ReID settings (see REID_BASE / REID_OPTIONS) and the result
is written to tuning_dev_reid.json; the no-ReID settings stay in that grid, so ReID can only be chosen when
it is better on the tuning segments.
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


# ReID grid, fixed in advance. Every base setting is also tried with ReID off (listed first, so ties go to
# the plain tracker), and with each (reid_weight, reid_gate, reid_thresh) option; ema is fixed at 0.9.
REID_BASE = {
    "score_thresh": [0.25, 0.5],
    "iou_threshold": [0.1, 0.2, 0.3],
    "max_age": [20, 40, 80],
    "min_hits": [1, 3],
}
REID_OPTIONS = [(0.3, 0.0, 0.4), (0.6, 0.0, 0.4), (0.0, 1.5, 0.25), (0.0, 1.5, 0.4),
                (0.3, 1.5, 0.25), (0.3, 1.5, 0.4), (0.6, 1.5, 0.25), (0.6, 1.5, 0.4)]
EXTENDED_GRID = {**GRID, "max_age": [5, 10, 20, 40, 80]}     # the existing grid plus max_age 80


def grid_settings(grid: dict, reid: bool = False) -> list[dict]:
    """All settings of `grid` (a dict of lists); with reid=True each is also tried with every ReID option."""
    base = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    if not reid:
        return base
    settings = list(base)
    for setting in base:
        for weight, gate, thresh in REID_OPTIONS:
            settings.append({**setting, "reid_weight": weight, "reid_gate": gate, "reid_thresh": thresh})
    return settings


def rank(frames: dict, class_names: list[str], settings: list[dict]) -> list[dict]:
    """Score every setting on `frames` and sort by weighted IDF1, ties by MOTA (stable, so earlier wins)."""
    rows = []
    for setting in settings:
        report = score(frames, class_names, **setting)
        rows.append({**setting, "mota": report["weighted_mota"], "idf1": report["weighted_idf1"],
                     "hota": report["weighted_hota"], "vehicle_switches": report["per_class"]["vehicle"]["num_switches"]})
    rows.sort(key=lambda r: (r["idf1"], r["mota"]), reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--dev-segments", type=int, default=3)
    parser.add_argument("--reid", action="store_true", help="Tune with the ReID grid; writes tuning_dev_reid.json.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(args.run_dir)
    class_names = cfg["model"].get("class_names", ["vehicle", "pedestrian", "cyclist", "sign"])
    toolkit = ToolKit(split="validation")
    eval_segments = load_split_manifest(cfg["data"]["manifest"])["eval"]
    segments = dev_segments(toolkit, eval_segments, args.dev_segments)
    print("dev segments:", *segments, sep="\n  ", flush=True)

    frames = collect(toolkit, segments, args.camera, cfg, run_dir, args.device)
    if args.reid:
        from modules.reid import attach_embeddings
        attach_embeddings(toolkit, frames, args.camera, run_dir, args.device)
        rows = rank(frames, class_names, grid_settings(REID_BASE, reid=True))
        out = run_dir / "tracking" / "tuning_dev_reid.json"
        out.write_text(json.dumps({"dev_segments": segments, "best": rows[0], "top5": rows[:5], "grid": rows}, indent=2) + "\n")
        for row in rows[:5]:
            print(row)
        return
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
