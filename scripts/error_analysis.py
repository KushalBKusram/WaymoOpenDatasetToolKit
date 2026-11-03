"""Slice a finished camera run's detection errors by object size and failure type.

Re-runs the checkpoint on the validation frames and writes error_analysis.json
plus side-by-side images of the worst frames next to the run's evaluation.json.

  python scripts/error_analysis.py --config configs/yolov8n.yaml \
      --run-dir runs/yolov8n-v1.2-s42 --device mps
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from evaluate import box_iou_xyxy, save_camera_sample  # noqa: E402
from models import build_detector  # noqa: E402
from modules.benchmark import load_split_manifest  # noqa: E402
from modules.runtime import resolve_torch_device  # noqa: E402
from modules.waymo_open_dataset import YOLO_CLASS_MAP, ToolKit, _C_BOX  # noqa: E402

SIZE_BINS = ("small", "medium", "large")   # COCO area ranges: <32^2, 32^2-96^2, >96^2 pixels
FP_TYPES = ("duplicate", "localization", "class_confusion", "background")


def size_bin(box) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


def analyse_frame(gts, dets, iou_match=0.5):
    """Greedy per-class matching; returns per-GT hit flags and per-detection outcomes."""
    gt_hit = [False] * len(gts)
    outcomes = []
    for det_index in sorted(range(len(dets)), key=lambda i: -dets[i]["score"]):
        det = dets[det_index]
        best_same = (0.0, -1)
        best_other = 0.0
        for gt_index, gt in enumerate(gts):
            overlap = box_iou_xyxy(det["box"], gt["box"])
            if gt["label"] == det["label"]:
                if not gt_hit[gt_index] and overlap > best_same[0]:
                    best_same = (overlap, gt_index)
            else:
                best_other = max(best_other, overlap)
        if best_same[0] >= iou_match:
            gt_hit[best_same[1]] = True
            outcomes.append((det_index, "tp"))
            continue
        same_any = max((box_iou_xyxy(det["box"], g["box"]) for g in gts if g["label"] == det["label"]),
                       default=0.0)
        if same_any >= iou_match:
            kind = "duplicate"
        elif best_other >= iou_match:
            kind = "class_confusion"
        elif same_any >= 0.1:
            kind = "localization"
        else:
            kind = "background"
        outcomes.append((det_index, kind))
    return gt_hit, outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--save-worst", type=int, default=12)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg.get("task") != "camera_2d_detection":
        raise SystemExit("error_analysis.py supports camera_2d_detection runs only.")
    run_dir = Path(args.run_dir)
    cfg["resume_weights"] = str(run_dir / "checkpoints" / "latest.pt")
    device = resolve_torch_device(args.device)
    detector = build_detector(cfg)
    model = detector.build_model(cfg, device).eval()
    class_names = cfg["model"].get("class_names", ["vehicle", "pedestrian", "cyclist", "sign"])
    toolkit = ToolKit(split="validation")
    manifest = cfg["data"].get("manifest")
    segments = (load_split_manifest(manifest)["eval"] if manifest else toolkit.list_segments())
    segments = segments[:args.segments or cfg.get("eval", {}).get("segs", 10)]

    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    gt_stats = {(c, b): [0, 0] for c in range(len(class_names)) for b in SIZE_BINS}   # [count, hit]
    fp_stats = {(c, k): 0 for c in range(len(class_names)) for k in FP_TYPES}
    fp_by_size = {b: 0 for b in SIZE_BINS}
    tp_total = 0
    frame_scores = []
    start = time.time()
    frames = 0

    for segment in segments:
        toolkit.assign_segment(segment)
        for timestamp in toolkit.get_timestamps():
            image = toolkit.load_camera_frame(timestamp, args.camera)
            rows = toolkit.load_camera_boxes(timestamp, args.camera)
            gts = []
            for _, row in rows.iterrows():
                label = YOLO_CLASS_MAP.get(int(row[f"{_C_BOX}.type"]))
                if label is None:
                    continue
                cx, cy = float(row[f"{_C_BOX}.box.center.x"]), float(row[f"{_C_BOX}.box.center.y"])
                w, h = float(row[f"{_C_BOX}.box.size.x"]), float(row[f"{_C_BOX}.box.size.y"])
                gts.append({"box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], "label": label})
            dets = detector.predict(model, image, cfg)
            gt_hit, outcomes = analyse_frame(gts, dets)
            for gt, hit in zip(gts, gt_hit):
                cell = gt_stats[(gt["label"], size_bin(gt["box"]))]
                cell[0] += 1
                cell[1] += int(hit)
            misses = sum(not h for h in gt_hit)
            false_positives = 0
            for det_index, kind in outcomes:
                if kind == "tp":
                    tp_total += 1
                else:
                    false_positives += 1
                    fp_stats[(dets[det_index]["label"], kind)] += 1
                    fp_by_size[size_bin(dets[det_index]["box"])] += 1
            frame_scores.append((misses + false_positives, segment, timestamp))
            metric.update(
                [{"boxes": torch.tensor([d["box"] for d in dets], dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.tensor([d["label"] for d in dets], dtype=torch.long),
                  "scores": torch.tensor([d["score"] for d in dets], dtype=torch.float32)}],
                [{"boxes": torch.tensor([g["box"] for g in gts], dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.tensor([g["label"] for g in gts], dtype=torch.long)}],
            )
            frames += 1
            if frames % 200 == 0:
                print(f"  {frames} frames  {time.time() - start:.0f}s", flush=True)
            if args.max_frames and frames >= args.max_frames:
                break
        if args.max_frames and frames >= args.max_frames:
            break

    coco = metric.compute()
    slices = []
    for c, name in enumerate(class_names):
        for b in SIZE_BINS:
            count, hit = gt_stats[(c, b)]
            slices.append({"class": name, "size": b, "ground_truth": count, "matched": hit,
                           "recall": round(hit / count, 4) if count else None})
    report = {
        "run": run_dir.name, "frames": frames, "match_iou": 0.5,
        "conf": cfg.get("eval", {}).get("conf", 0.25),
        "coco_map_by_size": {k: round(float(coco[k]), 4) for k in ("map", "map_small", "map_medium", "map_large")},
        "recall_by_class_and_size": slices,
        "true_positives": tp_total,
        "false_positives_by_type": {f"{class_names[c]}/{k}": v for (c, k), v in fp_stats.items() if v},
        "false_positives_by_size": fp_by_size,
    }
    (run_dir / "error_analysis.json").write_text(json.dumps(report, indent=2) + "\n")

    # Side-by-side images of the frames with the most errors.
    worst = sorted(frame_scores, reverse=True)[:args.save_worst]
    out_dir = run_dir / "error_analysis_worst"
    out_dir.mkdir(exist_ok=True)
    for errors, segment, timestamp in worst:
        toolkit.assign_segment(segment)
        image = toolkit.load_camera_frame(timestamp, args.camera)
        rows = toolkit.load_camera_boxes(timestamp, args.camera)
        save_camera_sample(image, rows, detector.predict(model, image, cfg),
                           out_dir / f"{errors:03d}errors_{segment[:12]}_{timestamp}.jpg")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
