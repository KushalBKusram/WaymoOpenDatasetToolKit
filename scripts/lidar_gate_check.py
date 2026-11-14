"""Sanity check of the per-box 3-D estimates behind LiDAR gating, against the LiDAR ground-truth boxes.

For every FRONT-camera ground-truth box the estimator (modules/lidar_gate.py) gives a vehicle-frame position from
the LiDAR points behind the box. Each is paired with the LiDAR ground-truth box of the same class whose projected
centre falls in the central part of the camera box (nearest to the box centre when several do), and the error of
the estimate against that box's centre is reported by class and range. The LiDAR centre is the middle of the
3-D box, while the points sit on the visible surface, so the depth error is expected to be biased towards the
sensor by about half the object's length; the spread around that bias is what matters for gating.

  python scripts/lidar_gate_check.py --out runs/lidar_gate_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from modules.benchmark import load_split_manifest  # noqa: E402
from modules.lidar_gate import camera_model_from_row, estimate_box, project_frame  # noqa: E402
from modules.waymo_open_dataset import LIDAR_DET_CLASS_MAP, ToolKit, _L_BOX  # noqa: E402
from track_eval import dev_segments, ground_truth  # noqa: E402

CLASSES = ("vehicle", "pedestrian", "cyclist")
RANGES = (("0-20 m", 0, 20), ("20-40 m", 20, 40), ("40+ m", 40, 1e9))


def project_point(xyz: np.ndarray, camera) -> tuple[float, float] | None:
    homogeneous = camera.vehicle_to_camera @ np.append(xyz, 1.0)
    depth = homogeneous[0]
    if depth <= 1e-3:
        return None
    return -homogeneous[1] / depth * camera.fu + camera.cu, -homogeneous[2] / depth * camera.fv + camera.cv


def summarize(errors: list[dict]) -> dict:
    out = {}
    for name in CLASSES:
        rows = [e for e in errors if e["class"] == name and e["estimated"]]
        total = [e for e in errors if e["class"] == name]
        if not total:
            continue
        entry = {"camera_boxes_matched_to_lidar_box": len(total),
                 "with_estimate": round(len(rows) / len(total), 3)}
        if rows:
            bev = np.array([e["bev"] for e in rows])
            depth = np.array([e["depth_bias"] for e in rows])
            entry.update({"median_bev_error_m": round(float(np.median(bev)), 2), "p90_bev_error_m": round(float(np.percentile(bev, 90)), 2),
                          "median_depth_bias_m": round(float(np.median(depth)), 2),
                          "depth_bias_spread_iqr_m": round(float(np.subtract(*np.percentile(depth, [75, 25]))), 2)})
        entry["by_range"] = {}
        for label, low, high in RANGES:
            in_range = [e for e in total if low <= e["range"] < high]
            with_est = [e for e in in_range if e["estimated"]]
            if in_range:
                entry["by_range"][label] = {"boxes": len(in_range), "with_estimate": round(len(with_est) / len(in_range), 3),
                                            "median_bev_error_m": round(float(np.median([e["bev"] for e in with_est])), 2) if with_est else None}
        out[name] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="configs/benchmark_splits.json")
    parser.add_argument("--segments", type=int, default=10)
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("runs/lidar_gate_check.json"))
    args = parser.parse_args()

    toolkit = ToolKit(split="validation")
    segments = load_split_manifest(args.manifest)["eval"][:args.segments]
    errors = []
    for segment in segments:
        toolkit.assign_segment(segment)
        camera = camera_model_from_row(toolkit.load_camera_calibration(1))
        for timestamp in sorted(int(t) for t in toolkit.get_timestamps()):
            gts = [g for g in ground_truth(toolkit, timestamp, 1) if g["label"] < 3]
            lidar_boxes = []
            for _, row in toolkit.load_lidar_boxes(timestamp).iterrows():
                class_id = LIDAR_DET_CLASS_MAP.get(int(row[f"{_L_BOX}.type"]))
                if class_id is not None and int(row[f"{_L_BOX}.num_top_lidar_points_in_box"]) >= 5:
                    lidar_boxes.append((class_id, np.array([float(row[f"{_L_BOX}.box.center.{k}"]) for k in "xyz"])))
            if not gts or not lidar_boxes:
                continue
            sets = toolkit.load_lidar_points_xyzi(timestamp, top_only=True)
            frame = project_frame(max(sets, key=len) if sets else np.zeros((0, 4)), camera)
            used = set()
            for gt in sorted(gts, key=lambda g: (g["box"][2] - g["box"][0]) * (g["box"][3] - g["box"][1])):
                x1, y1, x2, y2 = gt["box"]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                best, best_distance = None, None
                for index, (class_id, centre) in enumerate(lidar_boxes):
                    pixel = project_point(centre, camera)
                    if index in used or class_id != gt["label"] or pixel is None:
                        continue
                    if abs(pixel[0] - cx) <= 0.4 * (x2 - x1) and abs(pixel[1] - cy) <= 0.4 * (y2 - y1):
                        distance = np.hypot(pixel[0] - cx, pixel[1] - cy)
                        if best is None or distance < best_distance:
                            best, best_distance = index, distance
                if best is None:
                    continue
                used.add(best)
                centre = lidar_boxes[best][1]
                xyz, n = estimate_box(frame, gt["box"])
                record = {"class": CLASSES[gt["label"]], "range": float(np.hypot(centre[0], centre[1])),
                          "estimated": xyz is not None and n >= args.min_points}
                if record["estimated"]:
                    record["bev"] = float(np.hypot(*(xyz[:2] - centre[:2])))
                    record["depth_bias"] = float(np.hypot(*xyz[:2]) - np.hypot(centre[0], centre[1]))   # negative: nearer the sensor
                errors.append(record)
    result = {"segments": len(segments), "min_points": args.min_points, "pairs": len(errors), "summary": summarize(errors)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
