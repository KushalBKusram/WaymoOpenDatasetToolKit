"""Evaluate a checkpoint and persist metrics for the Streamlit run dashboard.

Examples:
  python evaluate.py --config configs/yolov8n.yaml --run-dir runs/yolov8n
  python evaluate.py --config configs/centerpoint.yaml --run-dir runs/centerpoint --max-frames 50

Camera reports use torchmetrics COCO mAP plus transparent IoU=0.5 error counts.
LiDAR reports remain explicitly labelled BEV-AABB proxy metrics, not official
Waymo rotated 3-D mAP.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from models import build_detector
from modules.run_artifacts import RunArtifacts
from modules.runtime import resolve_torch_device
from modules.visualize import draw_camera_boxes
from modules.waymo_open_dataset import (
    LIDAR_DET_CLASS_MAP,
    LIDAR_DET_CLASS_NAMES,
    YOLO_CLASS_MAP,
    ToolKit,
    _C_BOX,
    _L_BOX,
)


def box_iou_xyxy(a: list[float], b: list[float]) -> float:
    """Axis-aligned 2-D IoU for ``[x1, y1, x2, y2]`` boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def box_iou_aabb(a: list[float], b: list[float]) -> float:
    """Axis-aligned BEV IoU for ``[center_x, center_y, width, height]`` boxes."""
    return box_iou_xyxy(
        [a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2],
        [b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2],
    )


def average_precision(scored: list[tuple[float, int]], total_gt: int) -> float | None:
    """Integrate an interpolated-free precision/recall curve for proxy AP."""
    if not total_gt:
        return None
    ranked = sorted(scored, reverse=True)
    tp = np.cumsum([match for _, match in ranked])
    fp = np.cumsum([1 - match for _, match in ranked])
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.trapezoid(precision, recall))


def _counts(num_classes: int) -> dict[int, dict[str, int]]:
    return {class_id: {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
            for class_id in range(num_classes)}


def _match_by_class(
    gt_boxes: list[list[float]],
    gt_labels: list[int],
    detections: list[dict[str, Any]],
    iou_fn,
    threshold: float,
    num_classes: int,
) -> dict[int, dict[str, int]]:
    """Greedily match scored predictions to ground truth per class.

    These counts complement mAP with an auditable operating-point diagnostic;
    they are not used to compute the primary COCO metric.
    """
    result = _counts(num_classes)
    for class_id in range(num_classes):
        gt_indices = [index for index, label in enumerate(gt_labels) if label == class_id]
        predictions = sorted(
            (item for item in detections if int(item["label"]) == class_id),
            key=lambda item: float(item["score"]), reverse=True,
        )
        used: set[int] = set()
        for prediction in predictions:
            candidate = max(
                ((iou_fn(prediction["box"], gt_boxes[index]), index)
                 for index in gt_indices if index not in used),
                default=(0.0, -1),
            )
            if candidate[0] >= threshold:
                used.add(candidate[1])
                result[class_id]["true_positives"] += 1
            else:
                result[class_id]["false_positives"] += 1
        result[class_id]["false_negatives"] += len(gt_indices) - len(used)
    return result


def _merge_counts(target: dict[int, dict[str, int]], source: dict[int, dict[str, int]]) -> None:
    for class_id, values in source.items():
        for name, value in values.items():
            target[class_id][name] += int(value)


def _error_rows(counts: dict[int, dict[str, int]], class_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for class_id, values in counts.items():
        tp, fp, fn = values["true_positives"], values["false_positives"], values["false_negatives"]
        rows.append({
            "class_id": class_id,
            "class_name": class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
            **values,
            "precision": round(tp / (tp + fp), 6) if tp + fp else None,
            "recall": round(tp / (tp + fn), 6) if tp + fn else None,
        })
    return rows


def _latency_summary(latencies_seconds: list[float]) -> dict[str, float | int | None]:
    if not latencies_seconds:
        return {"samples": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "fps": None}
    milliseconds = np.asarray(latencies_seconds, dtype=np.float64) * 1000.0
    mean_ms = float(milliseconds.mean())
    return {
        "samples": int(len(milliseconds)),
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(float(np.median(milliseconds)), 3),
        "p95_ms": round(float(np.percentile(milliseconds, 95)), 3),
        "fps": round(1000.0 / mean_ms, 3) if mean_ms else None,
    }


def _camera_metric_rows(results: dict[str, Any], class_names: list[str]) -> list[dict[str, Any]]:
    classes = torch.as_tensor(results.get("classes", torch.empty(0, dtype=torch.long))).reshape(-1)
    values = torch.as_tensor(results.get("map_per_class", torch.empty(0))).reshape(-1)
    class_to_map = {
        int(class_id): (float(value) if float(value) >= 0.0 else None)
        for class_id, value in zip(classes.detach().cpu().tolist(), values.detach().cpu().tolist())
    }
    return [{
        "class_id": class_id,
        "class_name": name,
        "map": class_to_map.get(class_id),
    } for class_id, name in enumerate(class_names)]


def camera_report(
    results: dict[str, Any], frames: int, task: str, class_names: list[str],
    error_counts: dict[int, dict[str, int]], latencies: list[float], match_iou: float,
) -> dict[str, Any]:
    return {
        "task": task,
        "metric": "COCO mAP@[.50:.95]",
        "frames": frames,
        "map": float(results["map"]),
        "map_50": float(results["map_50"]),
        "map_75": float(results["map_75"]),
        "per_class": _camera_metric_rows(results, class_names),
        "error_analysis": {"match_iou": match_iou, "per_class": _error_rows(error_counts, class_names)},
        "inference": _latency_summary(latencies),
    }


def save_camera_sample(image: np.ndarray, gt_rows, detections: list[dict[str, Any]], path: Path) -> None:
    """Write side-by-side ground-truth and prediction review image."""
    ground_truth = draw_camera_boxes(image, gt_rows)
    prediction = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["box"])
        cv2.rectangle(prediction, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(prediction, f'{detection["label"]}: {detection["score"]:.2f}', (x1, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(ground_truth, "GROUND TRUTH", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(prediction, "PREDICTION", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.hconcat([ground_truth, prediction]))


def evaluate_camera(
    detector, model, cfg: dict[str, Any], toolkit: ToolKit, segments: list[str], cameras: list[int],
    max_frames: int | None, sample_dir: Path, sample_limit: int, fusion: bool, match_iou: float,
) -> dict[str, Any]:
    from torchmetrics.detection import MeanAveragePrecision

    class_names = cfg["model"].get("class_names", ["vehicle", "pedestrian", "cyclist", "sign"])
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    error_counts = _counts(len(class_names))
    latencies: list[float] = []
    frames = 0

    for segment in segments:
        toolkit.assign_segment(segment)
        for timestamp in toolkit.get_timestamps():
            for camera in cameras:
                image = toolkit.load_camera_frame(timestamp, camera)
                gt_rows = toolkit.load_camera_boxes(timestamp, camera)
                gt_boxes, gt_labels = [], []
                for _, row in gt_rows.iterrows():
                    class_id = YOLO_CLASS_MAP.get(int(row[f"{_C_BOX}.type"]))
                    if class_id is None:
                        continue
                    cx, cy = float(row[f"{_C_BOX}.box.center.x"]), float(row[f"{_C_BOX}.box.center.y"])
                    width, height = float(row[f"{_C_BOX}.box.size.x"]), float(row[f"{_C_BOX}.box.size.y"])
                    gt_boxes.append([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])
                    gt_labels.append(class_id)

                started = time.perf_counter()
                if fusion:
                    point_sets = toolkit.load_lidar_points_xyzi(timestamp)
                    if cfg["data"].get("top_lidar_only", True) and point_sets:
                        point_sets = [max(point_sets, key=len)]
                    points = (np.concatenate(point_sets, axis=0).astype(np.float32)
                              if point_sets else np.zeros((0, 4), np.float32))
                    detections = detector.predict(
                        model, image, cfg, lidar_points=points,
                        camera_calibration=toolkit.load_camera_calibration(camera),
                    )
                else:
                    detections = detector.predict(model, image, cfg)
                latencies.append(time.perf_counter() - started)

                if frames < sample_limit:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    save_camera_sample(image, gt_rows, detections, sample_dir / f"{segment}_{timestamp}_{camera}.jpg")

                _merge_counts(error_counts, _match_by_class(
                    gt_boxes, gt_labels, detections, box_iou_xyxy, match_iou, len(class_names)
                ))
                metric.update(
                    [{"boxes": torch.tensor([item["box"] for item in detections], dtype=torch.float32).reshape(-1, 4),
                      "labels": torch.tensor([item["label"] for item in detections], dtype=torch.long),
                      "scores": torch.tensor([item["score"] for item in detections], dtype=torch.float32)}],
                    [{"boxes": torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4),
                      "labels": torch.tensor(gt_labels, dtype=torch.long)}],
                )
                frames += 1
                if max_frames and frames >= max_frames:
                    return camera_report(metric.compute(), frames, cfg.get("task", "camera_2d_detection"),
                                         class_names, error_counts, latencies, match_iou)
    return camera_report(metric.compute(), frames, cfg.get("task", "camera_2d_detection"),
                         class_names, error_counts, latencies, match_iou)


def evaluate_lidar(
    detector, model, cfg: dict[str, Any], toolkit: ToolKit, segments: list[str],
    max_frames: int | None, match_iou: float,
) -> dict[str, Any]:
    thresholds = cfg.get("eval", {}).get("iou_thresholds", {})
    class_thresholds = [float(thresholds.get(name, default))
                        for name, default in zip(LIDAR_DET_CLASS_NAMES, [0.7, 0.5, 0.5])]
    scored = [[] for _ in LIDAR_DET_CLASS_NAMES]
    totals = [0 for _ in LIDAR_DET_CLASS_NAMES]
    error_counts = _counts(len(LIDAR_DET_CLASS_NAMES))
    prediction_counts = [0 for _ in LIDAR_DET_CLASS_NAMES]
    latencies: list[float] = []
    frames = 0

    for segment in segments:
        toolkit.assign_segment(segment)
        for timestamp in toolkit.get_timestamps():
            point_sets = toolkit.load_lidar_points_xyzi(timestamp)
            if cfg["data"].get("top_lidar_only", True) and point_sets:
                point_sets = [max(point_sets, key=len)]
            points = np.concatenate(point_sets, axis=0) if point_sets else np.zeros((0, 4), np.float32)
            started = time.perf_counter()
            detections = detector.predict(model, points, cfg)
            latencies.append(time.perf_counter() - started)
            gt_boxes, gt_labels = [], []
            for _, row in toolkit.load_lidar_boxes(timestamp).iterrows():
                class_id = LIDAR_DET_CLASS_MAP.get(int(row[f"{_L_BOX}.type"]))
                if class_id is None:
                    continue
                gt_boxes.append([
                    float(row[f"{_L_BOX}.box.center.x"]), float(row[f"{_L_BOX}.box.center.y"]),
                    float(row[f"{_L_BOX}.box.size.x"]), float(row[f"{_L_BOX}.box.size.y"]),
                ])
                gt_labels.append(class_id)

            for class_id in range(len(LIDAR_DET_CLASS_NAMES)):
                class_gt = [box for box, label in zip(gt_boxes, gt_labels) if label == class_id]
                class_detections = [item for item in detections if int(item["label"]) == class_id]
                prediction_counts[class_id] += len(class_detections)
                totals[class_id] += len(class_gt)
                used = [False] * len(class_gt)
                for item in sorted(class_detections, key=lambda value: value["score"], reverse=True):
                    box = item["box"]
                    candidate = [box[0], box[1], box[3], box[4]]
                    iou, index = max(
                        ((box_iou_aabb(candidate, target), index)
                         for index, target in enumerate(class_gt) if not used[index]),
                        default=(0.0, -1),
                    )
                    matched = iou >= class_thresholds[class_id]
                    if matched:
                        used[index] = True
                    scored[class_id].append((float(item["score"]), int(matched)))
            # The error analysis uses one explicit operating point. AP remains
            # the score-ranked curve above with Waymo-style class thresholds.
            _merge_counts(error_counts, _match_by_class(
                gt_boxes, gt_labels,
                [{"box": [item["box"][0], item["box"][1], item["box"][3], item["box"][4]],
                  "label": item["label"], "score": item["score"]} for item in detections],
                box_iou_aabb, match_iou, len(LIDAR_DET_CLASS_NAMES),
            ))
            frames += 1
            if max_frames and frames >= max_frames:
                break
        if max_frames and frames >= max_frames:
            break

    per_class_ap = [average_precision(items, total) for items, total in zip(scored, totals)]
    valid = [value for value in per_class_ap if value is not None]
    per_class = []
    errors = _error_rows(error_counts, LIDAR_DET_CLASS_NAMES)
    for class_id, name in enumerate(LIDAR_DET_CLASS_NAMES):
        per_class.append({
            "class_id": class_id, "class_name": name,
            "ap_proxy": per_class_ap[class_id], "ground_truth": totals[class_id],
            "predictions": prediction_counts[class_id], "iou_threshold": class_thresholds[class_id],
        })
    return {
        "task": "lidar_3d_detection",
        "metric": "BEV axis-aligned AP proxy (not official Waymo 3-D mAP)",
        "frames": frames,
        "map_proxy": float(np.mean(valid)) if valid else None,
        "per_class": per_class,
        "error_analysis": {"match_iou": match_iou, "per_class": errors},
        "inference": _latency_summary(latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Waymo toolkit checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default="./runs/waymo")
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--cameras", nargs="+", type=int, default=[1])
    parser.add_argument("--save-samples", type=int, default=12, metavar="N",
                        help="Number of camera prediction-review images to save.")
    parser.add_argument("--match-iou", type=float, default=0.5,
                        help="IoU operating point for precision/recall error counts (default: 0.5).")
    args = parser.parse_args()

    cfg, run_dir = yaml.safe_load(Path(args.config).read_text()), Path(args.run_dir)
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    cfg["resume_weights"] = str(checkpoint)
    device = resolve_torch_device(cfg.get("train", {}).get("device", "auto"))
    detector = build_detector(cfg)
    model = detector.build_model(cfg, device).eval()
    toolkit = ToolKit(split="validation")
    segments = toolkit.list_segments()[:args.segments or cfg.get("eval", {}).get("segs", 5)]
    task = cfg.get("task")
    report = (
        evaluate_lidar(detector, model, cfg, toolkit, segments, args.max_frames, args.match_iou)
        if task == "lidar_3d_detection" else
        evaluate_camera(detector, model, cfg, toolkit, segments, args.cameras, args.max_frames,
                        run_dir / "evaluation_samples", args.save_samples,
                        fusion=(task == "camera_lidar_fusion_2d_detection"), match_iou=args.match_iou)
    )
    report.update({"checkpoint": str(checkpoint), "segments": segments, "device": str(device)})
    artifacts = RunArtifacts(run_dir, cfg)
    artifacts.write_json("evaluation.json", report)
    artifacts.record("evaluation", **report)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
