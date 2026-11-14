"""Does LiDAR 3-D gating improve camera tracking? Fixed-protocol comparison over one or more detector runs.

Four trackers per detector run, each tuned on the 3 held-out dev segments with the rule fixed in advance (highest
ground-truth-weighted IDF1, ties by MOTA; the plain no-LiDAR settings are listed first so ties go to them) and each
scored once on the 10 evaluation segments:
  (a) baseline and (b) ReID: the settings chosen by scripts/reid_experiment.py on the same grids (stored in
      runs/<run>/tracking/reid_experiment.json); they are re-scored here as a consistency check;
  (c) LiDAR gating alone: LIDAR_BASE x LIDAR_OPTIONS (which also contains every no-LiDAR setting);
  (d) ReID + LiDAR gating: LIDAR_BASE x {no ReID, the ReID option chosen for (b)} x LIDAR_OPTIONS.
The evaluation segments are never used to choose settings.

  python scripts/lidar_gate_experiment.py --config configs/yolov8n_hires_100.yaml \\
      --run-dirs runs/yolov8n_hires_100-v1.2-s42 runs/yolov8n_hires_100-v1.2-s43 runs/yolov8n_hires_100-v1.2-s44
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from modules.benchmark import load_split_manifest  # noqa: E402
from modules.lidar_gate import attach_lidar3d  # noqa: E402
from modules.reid import attach_embeddings  # noqa: E402
from modules.waymo_open_dataset import ToolKit  # noqa: E402
from reid_experiment import evaluate  # noqa: E402
from track_eval import collect, dev_segments  # noqa: E402
from tune_tracker import REID_BASE, grid_settings, rank  # noqa: E402

LIDAR_BASE = REID_BASE
# (lidar_gate metres, lidar_weight, lidar_min_points), fixed in advance; the "off" setting is listed first.
LIDAR_OPTIONS = [(2.0, 0.0, 5), (4.0, 0.0, 5), (8.0, 0.0, 5), (4.0, 0.0, 15),
                 (0.0, 0.3, 5), (0.0, 0.6, 5), (4.0, 0.3, 5), (8.0, 0.3, 5)]
KEYS = ("score_thresh", "iou_threshold", "max_age", "min_hits", "reid_weight", "reid_gate", "reid_thresh",
        "lidar_gate", "lidar_weight", "lidar_min_points")


def clean(row: dict) -> dict:
    return {k: row[k] for k in KEYS if k in row}


def lidar_settings(grid: dict, reid_option: tuple | None = None) -> list[dict]:
    """Every base setting without LiDAR, then each with every LiDAR option (optionally with a fixed ReID option)."""
    settings = []
    reid = {} if reid_option is None else {"reid_weight": reid_option[0], "reid_gate": reid_option[1],
                                           "reid_thresh": reid_option[2]}
    for base in grid_settings(grid):
        settings.append({**base, **reid})
    for base in grid_settings(grid):
        for gate, weight, minimum in LIDAR_OPTIONS:
            settings.append({**base, **reid, "lidar_gate": gate, "lidar_weight": weight, "lidar_min_points": minimum})
    return settings


def coverage(frames: dict, class_names: list[str], minimum: int = 5) -> dict:
    """Fraction of detections (score >= 0.25) that get a 3-D estimate, by class and by box size."""
    rows = {}
    for label, name in enumerate(class_names):
        sides, has = [], []
        for segment_frames in frames.values():
            for _, dets in segment_frames:
                for det in dets:
                    if det["label"] == label and det["score"] >= 0.25:
                        sides.append(float(np.sqrt((det["box"][2] - det["box"][0]) * (det["box"][3] - det["box"][1]))))
                        has.append(det.get("xyz_world") is not None and det.get("n3d", 0) >= minimum)
        if sides:
            sides, has = np.array(sides), np.array(has)
            rows[name] = {"detections": int(len(sides)), "with_estimate": round(float(has.mean()), 3),
                          "by_box_size": {label_: round(float(has[mask].mean()), 3) if mask.any() else None
                                          for label_, mask in (("small<32px", sides < 32),
                                                               ("medium32-96px", (sides >= 32) & (sides < 96)),
                                                               ("large>=96px", sides >= 96))}}
    return rows


def run_one(args, run_dir: Path, toolkit: ToolKit, cfg: dict, class_names: list[str]) -> dict:
    eval_segments = load_split_manifest(cfg["data"]["manifest"])["eval"][:10]
    dev = dev_segments(toolkit, eval_segments, 3)
    started = time.time()
    dev_frames = collect(toolkit, dev, 1, cfg, run_dir, args.device)
    eval_frames = collect(toolkit, eval_segments, 1, cfg, run_dir, args.device)
    attach_embeddings(toolkit, dev_frames, 1, run_dir, args.device)
    attach_embeddings(toolkit, eval_frames, 1, run_dir, args.device)
    estimate_seconds = attach_lidar3d(toolkit, dev_frames, 1, run_dir) + attach_lidar3d(toolkit, eval_frames, 1, run_dir)

    previous = json.loads((run_dir / "tracking" / "reid_experiment.json").read_text())
    chosen = {"baseline": clean(previous["dev"]["baseline"]), "reid": clean(previous["dev"]["reid"])}
    reid_option = (chosen["reid"].get("reid_weight", 0.0), chosen["reid"].get("reid_gate", 0.0),
                   chosen["reid"].get("reid_thresh", 0.4))
    gate_rows = rank(dev_frames, class_names, lidar_settings(LIDAR_BASE))
    both_rows = rank(dev_frames, class_names, lidar_settings(LIDAR_BASE, reid_option) if any(reid_option[:2]) else [])
    chosen["lidar"] = clean(gate_rows[0])
    if both_rows:
        chosen["reid_lidar"] = clean(both_rows[0])
    dev_scores = {name: next(r for r in (gate_rows if name == "lidar" else both_rows) if clean(r) == chosen[name])
                  for name in ("lidar", "reid_lidar") if name in chosen}

    result = {"run": run_dir.name, "dev_segments": dev, "estimate_seconds": round(estimate_seconds, 1),
              "chosen": chosen, "reid_option": reid_option,
              "dev": {name: {"idf1": row["idf1"], "mota": row["mota"], "hota": row["hota"]} for name, row in dev_scores.items()},
              "dev_top5_lidar": [{**clean(r), "idf1": r["idf1"], "mota": r["mota"]} for r in gate_rows[:5]],
              "coverage_eval": coverage(eval_frames, class_names),
              "eval": {name: evaluate(eval_frames, class_names, settings) for name, settings in chosen.items()},
              "seconds": round(time.time() - started, 1)}
    out = run_dir / "tracking" / "lidar_gate_experiment.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def summarize(results: list[dict]) -> dict:
    names = [n for n in ("baseline", "reid", "lidar", "reid_lidar") if all(n in r["eval"] for r in results)]
    summary = {"runs": [r["run"] for r in results], "metrics": {}, "minus_baseline": {}}
    for name in names:
        summary["metrics"][name] = {}
        for key in ("mota", "idf1", "hota", "deta", "assa"):
            values = [r["eval"][name]["weighted"][key] for r in results]
            summary["metrics"][name][key] = {"values": values, "mean": float(np.mean(values)),
                                              "min": float(min(values)), "max": float(max(values))}
        for key in ("total_switches", "total_switches_after_gap"):
            values = [r["eval"][name][key] for r in results]
            summary["metrics"][name][key] = {"values": values, "mean": float(np.mean(values))}
        if name != "baseline":
            summary["minus_baseline"][name] = {
                key: [round(r["eval"][name]["weighted"][key] - r["eval"]["baseline"]["weighted"][key], 4) for r in results]
                for key in ("mota", "idf1", "hota", "deta", "assa")}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("runs/lidar_gate_experiment_summary.json"))
    parser.add_argument("--summarize-only", action="store_true",
                        help="Combine the per-run lidar_gate_experiment.json files written by earlier invocations.")
    args = parser.parse_args()
    if args.summarize_only:
        results = [json.loads((Path(run) / "tracking" / "lidar_gate_experiment.json").read_text()) for run in args.run_dirs]
        summary = summarize(results)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary["minus_baseline"], indent=1))
        return
    cfg = yaml.safe_load(Path(args.config).read_text())
    class_names = cfg["model"].get("class_names", ["vehicle", "pedestrian", "cyclist", "sign"])
    toolkit = ToolKit(split="validation")
    results = []
    for run in args.run_dirs:
        print(f"=== {run}", flush=True)
        result = run_one(args, Path(run), toolkit, dict(cfg), class_names)
        results.append(result)
        print(json.dumps({k: result[k] for k in ("chosen", "dev", "estimate_seconds", "seconds")}, indent=1), flush=True)
    summary = summarize(results)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["minus_baseline"], indent=1))


if __name__ == "__main__":
    main()
