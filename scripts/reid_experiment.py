"""Does appearance ReID improve camera tracking? Fixed-protocol comparison over one or more detector runs.

For each detector run (for example the three YOLOv8n setting-L seeds):
  1. tune the plain tracker (the existing grid) and the ReID tracker (REID_BASE x REID_OPTIONS, which also
     contains every no-ReID setting) on the 3 held-out dev segments, rule fixed in advance: highest
     ground-truth-weighted IDF1, ties by MOTA;
  2. score each chosen setting once on the 10 evaluation segments (MOTA, IDF1, HOTA, ID switches, and the
     ID switches that happen after a gap of more than one frame).
A third row, the plain tracker tuned on the extended grid (max_age up to 80, as the ReID grid allows),
separates "longer track memory" from "appearance".

  python scripts/reid_experiment.py --config configs/yolov8n_hires_100.yaml \\
      --run-dirs runs/yolov8n_hires_100-v1.2-s42 runs/yolov8n_hires_100-v1.2-s43 runs/yolov8n_hires_100-v1.2-s44
"""

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
from modules.reid import attach_embeddings  # noqa: E402
from modules.waymo_open_dataset import ToolKit  # noqa: E402
from track_eval import collect, dev_segments, score  # noqa: E402
from track_replay import replay_segment  # noqa: E402
from tune_tracker import EXTENDED_GRID, GRID, REID_BASE, grid_settings, rank  # noqa: E402

SETTING_KEYS = ("score_thresh", "iou_threshold", "max_age", "min_hits", "reid_weight", "reid_gate", "reid_thresh")


def clean(row: dict) -> dict:
    return {k: row[k] for k in SETTING_KEYS if k in row}


def evaluate(frames: dict, class_names: list[str], settings: dict) -> dict:
    """One scoring of the evaluation segments plus the replay counts of ID switches (all, and after a gap)."""
    report = score(frames, class_names, **settings)
    switches = {name: {"switches": 0, "after_gap": 0, "fragmentations": 0} for name in class_names}
    for segment_frames in frames.values():
        _, events = replay_segment(segment_frames, class_names, settings, 0.5)
        for event in events:
            if event["type"] == "switch":
                switches[event["class"]]["switches"] += 1
                switches[event["class"]]["after_gap"] += int(event["gap"] > 1)
            elif event["type"] == "fragment":
                switches[event["class"]]["fragmentations"] += 1
    keys = ("mota", "idf1", "hota", "deta", "assa", "num_switches", "num_fragmentations")
    return {"settings": report["settings"],
            "weighted": {k: report[f"weighted_{k}"] for k in ("mota", "idf1", "hota", "deta", "assa")},
            "per_class": {n: {k: v[k] for k in keys} for n, v in report["per_class"].items()},
            "replay": switches,
            "total_switches": sum(v["switches"] for v in switches.values()),
            "total_switches_after_gap": sum(v["after_gap"] for v in switches.values())}


def run_one(args, run_dir: Path, toolkit: ToolKit, cfg: dict, class_names: list[str]) -> dict:
    eval_segments = load_split_manifest(cfg["data"]["manifest"])["eval"][:10]
    dev = dev_segments(toolkit, eval_segments, 3)
    started = time.time()
    dev_frames = collect(toolkit, dev, 1, cfg, run_dir, args.device)
    eval_frames = collect(toolkit, eval_segments, 1, cfg, run_dir, args.device)
    embed_seconds = attach_embeddings(toolkit, dev_frames, 1, run_dir, args.device) \
        + attach_embeddings(toolkit, eval_frames, 1, run_dir, args.device)

    extended = rank(dev_frames, class_names, grid_settings(EXTENDED_GRID))
    baseline = next(r for r in extended if r["max_age"] <= max(GRID["max_age"]))          # the existing grid
    reid_rows = rank(dev_frames, class_names, grid_settings(REID_BASE, reid=True))
    chosen = {"baseline": baseline, "baseline_extended": extended[0], "reid": reid_rows[0]}

    result = {"run": run_dir.name, "dev_segments": dev, "embed_seconds": round(embed_seconds, 1),
              "dev": {name: {**clean(row), "mota": row["mota"], "idf1": row["idf1"], "hota": row["hota"]}
                      for name, row in chosen.items()},
              "dev_top5_reid": [{**clean(r), "idf1": r["idf1"], "mota": r["mota"]} for r in reid_rows[:5]],
              "eval": {name: evaluate(eval_frames, class_names, clean(row)) for name, row in chosen.items()},
              "seconds": round(time.time() - started, 1)}
    out = run_dir / "tracking" / "reid_experiment.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def summarize(results: list[dict]) -> dict:
    """Mean, range and per-seed ReID-minus-baseline differences of the headline metrics."""
    summary = {"runs": [r["run"] for r in results], "metrics": {}}
    for name in ("baseline", "baseline_extended", "reid"):
        summary["metrics"][name] = {}
        for key in ("mota", "idf1", "hota", "deta", "assa"):
            values = [r["eval"][name]["weighted"][key] for r in results]
            summary["metrics"][name][key] = {"values": values, "mean": float(np.mean(values)),
                                              "min": float(min(values)), "max": float(max(values))}
        for key in ("total_switches", "total_switches_after_gap"):
            values = [r["eval"][name][key] for r in results]
            summary["metrics"][name][key] = {"values": values, "mean": float(np.mean(values))}
    summary["reid_minus_baseline"] = {
        key: [round(r["eval"]["reid"]["weighted"][key] - r["eval"]["baseline"]["weighted"][key], 4) for r in results]
        for key in ("mota", "idf1", "hota", "deta", "assa")}
    summary["reid_minus_baseline_extended"] = {
        key: [round(r["eval"]["reid"]["weighted"][key] - r["eval"]["baseline_extended"]["weighted"][key], 4) for r in results]
        for key in ("mota", "idf1", "hota", "deta", "assa")}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("runs/reid_experiment_summary.json"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    class_names = cfg["model"].get("class_names", ["vehicle", "pedestrian", "cyclist", "sign"])
    toolkit = ToolKit(split="validation")
    results = []
    for run in args.run_dirs:
        print(f"=== {run}", flush=True)
        result = run_one(args, Path(run), toolkit, dict(cfg), class_names)
        results.append(result)
        print(json.dumps({k: result[k] for k in ("embed_seconds", "dev", "seconds")}, indent=1), flush=True)
    summary = summarize(results)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["reid_minus_baseline"], indent=1))


if __name__ == "__main__":
    main()
