"""Camera tracking-by-detection baseline: detector + SORT-style tracker, scored with MOT metrics.

Runs the detector over each validation segment in time order, tracks the detections, and
compares the tracks with the FRONT camera's ground-truth object ids. Detections are
cached in the run directory so tracker settings can be tuned without re-running the model.

  python scripts/track_eval.py --config configs/yolov8n.yaml --run-dir runs/yolov8n-v1.2-s42
  python scripts/track_eval.py --config configs/yolov8n.yaml --run-dir runs/yolov8n-v1.2-s42 --oracle

`--identity-oracle` skips the tracker and must score 1.0 on every metric (checks the scoring code).
`--oracle` feeds ground-truth boxes to the tracker as detections: it checks the pipeline and
bounds what the tracker alone can score.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import motmetrics as mm  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from models import build_detector  # noqa: E402
from modules.benchmark import load_split_manifest  # noqa: E402
from modules.hota import combine as hota_combine, hota_sequence, summarize as hota_summarize  # noqa: E402
from modules.runtime import resolve_torch_device  # noqa: E402
from modules.tracking import SortTracker, iou_matrix  # noqa: E402
from modules.waymo_open_dataset import YOLO_CLASS_MAP, ToolKit, _C_BOX  # noqa: E402

METRICS = ["num_frames", "num_objects", "mota", "idf1", "precision", "recall", "num_switches",
           "num_fragmentations", "mostly_tracked", "mostly_lost", "num_false_positives", "num_misses"]


def ground_truth(toolkit: ToolKit, timestamp: int, camera: int) -> list[dict]:
    gts = []
    for _, row in toolkit.load_camera_boxes(timestamp, camera).iterrows():
        label = YOLO_CLASS_MAP.get(int(row[f"{_C_BOX}.type"]))
        if label is None:
            continue
        cx, cy = float(row[f"{_C_BOX}.box.center.x"]), float(row[f"{_C_BOX}.box.center.y"])
        w, h = float(row[f"{_C_BOX}.box.size.x"]), float(row[f"{_C_BOX}.box.size.y"])
        gts.append({"id": str(row["key.camera_object_id"]), "label": label,
                    "box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]})
    return gts


def match_distances(gts: list[dict], tracks: list[dict], min_iou: float) -> np.ndarray:
    """1 - IoU for pairs overlapping at least `min_iou`; NaN (not matchable) otherwise."""
    overlap = iou_matrix(np.array([g["box"] for g in gts], dtype=float).reshape(-1, 4),
                         np.array([t["box"] for t in tracks], dtype=float).reshape(-1, 4))
    return np.where(overlap >= min_iou, 1.0 - overlap, np.nan)


def collect(toolkit: ToolKit, segments: list[str], camera: int, cfg: dict, run_dir: Path, device: str,
            oracle: bool = False) -> dict[str, list[tuple[list[dict], list[dict]]]]:
    """Per segment, in time order: (ground truth, all detections). Detections come from a cache in the run directory."""
    cache_path = run_dir / "tracking" / "detections.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() and not oracle else {}
    detector = model = None
    frames: dict[str, list[tuple[list[dict], list[dict]]]] = {}
    started, count = time.time(), 0
    for segment in segments:
        toolkit.assign_segment(segment)
        frames[segment] = []
        for timestamp in sorted(int(t) for t in toolkit.get_timestamps()):
            gts = ground_truth(toolkit, timestamp, camera)
            if oracle:
                dets = [{"box": g["box"], "label": g["label"], "score": 1.0} for g in gts]
            else:
                key = f"{segment}/{timestamp}"
                if key not in cache:
                    if detector is None:
                        cfg["resume_weights"] = str(run_dir / "checkpoints" / "latest.pt")
                        detector = build_detector(cfg)
                        model = detector.build_model(cfg, resolve_torch_device(device)).eval()
                    image = toolkit.load_camera_frame(timestamp, camera)
                    cache[key] = [{"box": [float(v) for v in d["box"]], "label": int(d["label"]),
                                   "score": float(d["score"])} for d in detector.predict(model, image, cfg)]
                dets = cache[key]
            frames[segment].append((gts, dets))
            count += 1
            if count % 200 == 0:
                print(f"  {count} frames  {time.time() - started:.0f}s", flush=True)
    if not oracle and detector is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    return frames


def score(frames: dict, class_names: list[str], score_thresh: float = 0.25, iou_threshold: float = 0.3,
          max_age: int = 5, min_hits: int = 2, match_iou: float = 0.5, identity_oracle: bool = False,
          reid_weight: float = 0.0, reid_gate: float = 0.0, reid_thresh: float = 0.4, ema: float = 0.9,
          lidar_gate: float = 0.0, lidar_weight: float = 0.0, lidar_min_points: int = 0) -> dict:
    """Track every segment and return pooled per-class MOT metrics plus ground-truth-weighted MOTA / IDF1.

    HOTA (modules/hota.py) is reported per class and as weighted_hota / weighted_deta / weighted_assa, with the
    same ground-truth-box weights as MOTA and IDF1. ReID settings need detections that carry an 'embedding'
    (modules.reid.attach_embeddings); LiDAR gating settings need detections that carry 'xyz_world' / 'n3d'
    (modules.lidar_gate.attach_lidar3d); with the defaults the tracker is the plain IoU tracker.
    """
    accs = {name: [] for name in class_names}
    hota_runs: dict[str, list] = {name: [] for name in class_names}      # per class, one frame list per segment
    for segment, segment_frames in frames.items():
        tracker = SortTracker(iou_threshold, max_age, min_hits, reid_weight=reid_weight, reid_gate=reid_gate,
                              reid_thresh=reid_thresh, ema=ema, lidar_gate=lidar_gate, lidar_weight=lidar_weight,
                              lidar_min_points=lidar_min_points)
        per_class = {name: mm.MOTAccumulator(auto_id=True) for name in class_names}
        gt_ids: dict[str, int] = {}     # motmetrics needs numeric object ids
        hota_frames: dict[str, list] = {name: [] for name in class_names}
        for gts, dets in segment_frames:
            if identity_oracle:
                tracks = [{"id": gt_ids.setdefault(g["id"], len(gt_ids)), "box": g["box"], "label": g["label"]}
                          for g in gts]
            else:
                tracks = tracker.update([d for d in dets if d["score"] >= score_thresh])
            for label, name in enumerate(class_names):
                g = [x for x in gts if x["label"] == label]
                t = [x for x in tracks if x["label"] == label]
                per_class[name].update([gt_ids.setdefault(x["id"], len(gt_ids)) for x in g], [x["id"] for x in t],
                                       match_distances(g, t, match_iou))
                hota_frames[name].append(([x["id"] for x in g], [x["box"] for x in g],
                                          [x["id"] for x in t], [x["box"] for x in t]))
        for name in class_names:
            accs[name].append(per_class[name])
            hota_runs[name].append(hota_frames[name])

    handler = mm.metrics.create()
    report = {"frames": sum(len(v) for v in frames.values()), "segments": list(frames),
              "settings": {"score_thresh": score_thresh, "iou_threshold": iou_threshold, "max_age": max_age,
                           "min_hits": min_hits, "match_iou": match_iou},
              "per_class": {}}
    if reid_weight > 0 or reid_gate > 0:
        report["settings"].update({"reid_weight": reid_weight, "reid_gate": reid_gate,
                                   "reid_thresh": reid_thresh, "ema": ema})
    if lidar_gate > 0 or lidar_weight > 0:
        report["settings"].update({"lidar_gate": lidar_gate, "lidar_weight": lidar_weight,
                                   "lidar_min_points": lidar_min_points})
    # Pool every segment per class: identities never cross segment boundaries.
    for name in class_names:
        row = handler.compute_many(accs[name], metrics=METRICS, names=list(frames), generate_overall=True).loc["OVERALL"]
        if row["num_objects"] == 0:
            continue
        report["per_class"][name] = {k: (round(float(row[k]), 4) if isinstance(row[k], (float, np.floating)) else int(row[k]))
                                     for k in METRICS}
        hota = hota_summarize(hota_combine([hota_sequence(frames_) for frames_ in hota_runs[name]]))
        report["per_class"][name].update({
            "hota": round(hota["HOTA"], 4), "deta": round(hota["DetA"], 4), "assa": round(hota["AssA"], 4),
            "deta_recall": round(hota["DetRe"], 4), "deta_precision": round(hota["DetPr"], 4),
            "assa_recall": round(hota["AssRe"], 4), "assa_precision": round(hota["AssPr"], 4),
            "loca": round(hota["LocA"], 4)})
    # Class-agnostic headline: weight per-class counts by ground-truth objects.
    total = sum(v["num_objects"] for v in report["per_class"].values())
    if total:
        report["weighted_mota"] = round(sum(v["mota"] * v["num_objects"] for v in report["per_class"].values()) / total, 4)
        report["weighted_idf1"] = round(sum(v["idf1"] * v["num_objects"] for v in report["per_class"].values()) / total, 4)
        for key in ("hota", "deta", "assa"):
            report[f"weighted_{key}"] = round(sum(v[key] * v["num_objects"] for v in report["per_class"].values()) / total, 4)
    return report


def dev_segments(toolkit: ToolKit, eval_segments: list[str], count: int) -> list[str]:
    """Validation segments after the frozen eval set (never reported), for tuning tracker settings."""
    return [s for s in sorted(toolkit.list_segments()) if s not in set(eval_segments)][:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--score-thresh", type=float, default=0.25, help="Minimum detection score given to the tracker.")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="Tracker association IoU.")
    parser.add_argument("--max-age", type=int, default=5)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU for a track to count as matching ground truth.")
    parser.add_argument("--reid-weight", type=float, default=0.0, help="Cosine-distance weight in the association cost.")
    parser.add_argument("--reid-gate", type=float, default=0.0, help="Re-link radius factor (0 = no re-link stage).")
    parser.add_argument("--reid-thresh", type=float, default=0.4, help="Maximum cosine distance for a re-link.")
    parser.add_argument("--ema", type=float, default=0.9, help="Weight of the old embedding in a track's average.")
    parser.add_argument("--lidar-gate", type=float, default=0.0, help="LiDAR 3-D gate in metres (0 = off).")
    parser.add_argument("--lidar-weight", type=float, default=0.0, help="Weight of the 3-D distance in the association cost.")
    parser.add_argument("--lidar-min-points", type=int, default=0, help="Ignore 3-D estimates from fewer LiDAR points.")
    parser.add_argument("--oracle", action="store_true", help="Use ground-truth boxes as detections.")
    parser.add_argument("--identity-oracle", action="store_true",
                        help="Give every ground-truth box its true id as a track: metrics must be perfect (code check).")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(args.run_dir)
    class_names = cfg["model"].get("class_names", ["vehicle", "pedestrian", "cyclist", "sign"])
    toolkit = ToolKit(split="validation")
    manifest = cfg["data"].get("manifest")
    segments = load_split_manifest(manifest)["eval"] if manifest else toolkit.list_segments()
    segments = segments[:args.segments or cfg.get("eval", {}).get("segs", 10)]

    frames = collect(toolkit, segments, args.camera, cfg, run_dir, args.device, oracle=args.oracle)
    if args.reid_weight > 0 or args.reid_gate > 0:
        from modules.reid import attach_embeddings
        spent = attach_embeddings(toolkit, frames, args.camera, run_dir, args.device)
        print(f"embeddings ready ({spent:.0f}s embedding)", flush=True)
    if args.lidar_gate > 0 or args.lidar_weight > 0:
        from modules.lidar_gate import attach_lidar3d
        spent = attach_lidar3d(toolkit, frames, args.camera, run_dir)
        print(f"3-D estimates ready ({spent:.0f}s)", flush=True)
    report = score(frames, class_names, args.score_thresh, args.iou_threshold, args.max_age, args.min_hits,
                   args.match_iou, args.identity_oracle, args.reid_weight, args.reid_gate, args.reid_thresh, args.ema,
                   args.lidar_gate, args.lidar_weight, args.lidar_min_points)
    report = {"run": run_dir.name, "oracle": args.oracle or args.identity_oracle, **report}
    kind = "identity_oracle" if args.identity_oracle else "oracle" if args.oracle else None
    out = args.out or run_dir / "tracking" / (f"tracking_{kind}.json" if kind else "tracking.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
