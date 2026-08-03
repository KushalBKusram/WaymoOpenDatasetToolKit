"""Evaluate a checkpoint and write a portable report for the Streamlit dashboard.

Examples:
  python evaluate.py --config configs/yolov8n.yaml --run-dir runs/waymo
  python evaluate.py --config configs/pointpillars.yaml --run-dir runs/waymo --max-frames 50

Camera reports use torchmetrics COCO-style 2-D mAP. LiDAR reports are clearly
labelled BEV-AABB proxy metrics; use the official Waymo evaluator for benchmark
or competition claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from models import build_detector
from modules.run_artifacts import RunArtifacts
from modules.runtime import resolve_torch_device
from modules.visualize import draw_camera_boxes
from modules.waymo_open_dataset import LIDAR_DET_CLASS_MAP, YOLO_CLASS_MAP, ToolKit, _C_BOX, _L_BOX


def box_iou_aabb(a: list[float], b: list[float]) -> float:
    """Axis-aligned 2-D IoU for [cx, cy, width, height] boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union else 0.0


def average_precision(scored: list[tuple[float, int]], total_gt: int) -> float | None:
    if not total_gt:
        return None
    scored.sort(reverse=True)
    tp = np.cumsum([match for _, match in scored])
    fp = np.cumsum([1 - match for _, match in scored])
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.trapz(precision, recall))


def camera_report(results: dict, frames: int, task: str = "camera_2d_detection") -> dict:
    return {"task": task, "metric": "COCO mAP@[.50:.95]", "frames": frames,
            "map": float(results["map"]), "map_50": float(results["map_50"]),
            "map_75": float(results["map_75"]), "per_class_map": results.get("map_per_class", torch.empty(0)).tolist()}


def save_camera_sample(image: np.ndarray, gt_rows, detections: list[dict], path: Path) -> None:
    """Write side-by-side ground-truth and prediction review image."""
    ground_truth = draw_camera_boxes(image, gt_rows)
    prediction = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["box"])
        cv2.rectangle(prediction, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(prediction, f'{detection["label"]}: {detection["score"]:.2f}', (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(ground_truth, "GROUND TRUTH", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(prediction, "PREDICTION", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.hconcat([ground_truth, prediction]))


def evaluate_camera(detector, model, cfg: dict, toolkit: ToolKit, segments: list[str], cameras: list[int], max_frames: int | None, sample_dir: Path, sample_limit: int, fusion: bool = False) -> dict:
    from torchmetrics.detection import MeanAveragePrecision
    metric, frames = MeanAveragePrecision(iou_type="bbox", box_format="xyxy"), 0
    for segment in segments:
        toolkit.assign_segment(segment)
        for ts in toolkit.get_timestamps():
            for camera in cameras:
                image, gt_rows = toolkit.load_camera_frame(ts, camera), toolkit.load_camera_boxes(ts, camera)
                gt_boxes, gt_labels = [], []
                for _, row in gt_rows.iterrows():
                    class_id = YOLO_CLASS_MAP.get(int(row[f"{_C_BOX}.type"]))
                    if class_id is None:
                        continue
                    cx, cy = float(row[f"{_C_BOX}.box.center.x"]), float(row[f"{_C_BOX}.box.center.y"])
                    w, h = float(row[f"{_C_BOX}.box.size.x"]), float(row[f"{_C_BOX}.box.size.y"])
                    gt_boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]); gt_labels.append(class_id)
                if fusion:
                    point_sets = toolkit.load_lidar_points_xyzi(ts)
                    if cfg['data'].get('top_lidar_only', True) and point_sets:
                        point_sets = [max(point_sets, key=len)]
                    points = (np.concatenate(point_sets, axis=0).astype(np.float32)
                              if point_sets else np.zeros((0, 4), np.float32))
                    detections = detector.predict(
                        model, image, cfg, lidar_points=points,
                        camera_calibration=toolkit.load_camera_calibration(camera),
                    )
                else:
                    detections = detector.predict(model, image, cfg)
                if frames < sample_limit:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    save_camera_sample(image, gt_rows, detections, sample_dir / f"{segment}_{ts}_{camera}.jpg")
                metric.update([{"boxes": torch.tensor([d["box"] for d in detections], dtype=torch.float32).reshape(-1, 4),
                                "labels": torch.tensor([d["label"] for d in detections], dtype=torch.long),
                                "scores": torch.tensor([d["score"] for d in detections], dtype=torch.float32)}],
                              [{"boxes": torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4),
                                "labels": torch.tensor(gt_labels, dtype=torch.long)}])
                frames += 1
                if max_frames and frames >= max_frames:
                    return camera_report(metric.compute(), frames, cfg.get("task", "camera_2d_detection"))
    return camera_report(metric.compute(), frames, cfg.get("task", "camera_2d_detection"))


def evaluate_lidar(detector, model, cfg: dict, toolkit: ToolKit, segments: list[str], max_frames: int | None) -> dict:
    thresholds, scored, total_gt, frames = [0.7, 0.5, 0.5], [[], [], []], [0, 0, 0], 0
    for segment in segments:
        toolkit.assign_segment(segment)
        for ts in toolkit.get_timestamps():
            chunks = toolkit.load_lidar_points_xyzi(ts)
            pred = detector.predict(model, np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 4), np.float32), cfg)
            gt_by_class = [[], [], []]
            for _, row in toolkit.load_lidar_boxes(ts).iterrows():
                cls = LIDAR_DET_CLASS_MAP.get(int(row[f"{_L_BOX}.type"]))
                if cls is not None:
                    gt_by_class[cls].append([float(row[f"{_L_BOX}.box.center.x"]), float(row[f"{_L_BOX}.box.center.y"]), float(row[f"{_L_BOX}.box.size.x"]), float(row[f"{_L_BOX}.box.size.y"])])
            for cls, gt_boxes in enumerate(gt_by_class):
                total_gt[cls] += len(gt_boxes); used = [False] * len(gt_boxes)
                for item in sorted((d for d in pred if d["label"] == cls), key=lambda d: d["score"], reverse=True):
                    b = item["box"]; candidate = [b[0], b[1], b[3], b[4]]
                    iou, index = max(((box_iou_aabb(candidate, gt), i) for i, gt in enumerate(gt_boxes) if not used[i]), default=(0.0, -1))
                    match = iou >= thresholds[cls]
                    if match: used[index] = True
                    scored[cls].append((float(item["score"]), int(match)))
            frames += 1
            if max_frames and frames >= max_frames: break
        if max_frames and frames >= max_frames: break
    ap = [average_precision(items, total) for items, total in zip(scored, total_gt)]
    valid = [value for value in ap if value is not None]
    return {"task": "lidar_3d_detection", "metric": "BEV axis-aligned AP proxy (not official Waymo 3-D mAP)", "frames": frames,
            "per_class_ap": ap, "map_proxy": float(np.mean(valid)) if valid else None,
            "ground_truth_per_class": total_gt, "iou_thresholds": thresholds}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Waymo toolkit checkpoint.")
    parser.add_argument("--config", required=True); parser.add_argument("--run-dir", default="./runs/waymo")
    parser.add_argument("--segments", type=int, default=None); parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--cameras", nargs="+", type=int, default=[1])
    parser.add_argument("--save-samples", type=int, default=12, metavar="N", help="Number of camera prediction-review images to save.")
    args = parser.parse_args()
    cfg, run_dir = yaml.safe_load(Path(args.config).read_text()), Path(args.run_dir)
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    if not checkpoint.exists(): raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    cfg["resume_weights"] = str(checkpoint)
    device = resolve_torch_device(cfg.get('train', {}).get('device', 'auto'))
    detector = build_detector(cfg); model = detector.build_model(cfg, device).eval(); toolkit = ToolKit(split="validation")
    segments = toolkit.list_segments()[:args.segments or cfg.get("eval", {}).get("segs", 5)]
    task = cfg.get("task")
    report = (evaluate_lidar(detector, model, cfg, toolkit, segments, args.max_frames)
              if task == "lidar_3d_detection" else
              evaluate_camera(detector, model, cfg, toolkit, segments, args.cameras,
                              args.max_frames, run_dir / "evaluation_samples",
                              args.save_samples,
                              fusion=(task == "camera_lidar_fusion_2d_detection")))
    report.update({"checkpoint": str(checkpoint), "segments": segments, "device": str(device)})
    artifacts = RunArtifacts(run_dir, cfg); artifacts.write_json("evaluation.json", report); artifacts.record("evaluation", **report)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__": main()
