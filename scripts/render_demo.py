"""Render a short side-by-side demo clip: camera detection + tracking, and a LiDAR bird's-eye view.

Left panel: FRONT camera with YOLOv8n detections tracked by the SORT-style tracker (coloured `#id` boxes).
Right panel: TOP-LiDAR points from above with the ground-truth 3-D boxes (green) and LiDAR detector
predictions (orange). The two panels are different models on the same timestamps; this is not fusion.

  # find clip windows with tracking events in an evaluation segment
  python scripts/render_demo.py --suggest

  # render one
  python scripts/render_demo.py --segment <context_name> --start 40 --frames 90 --out demo.mp4

Frames are Waymo Open Dataset data: keep clips short (illustration only) and keep the attribution
that the renderer draws, per https://waymo.com/open/terms.
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from models import build_detector  # noqa: E402
from modules.benchmark import load_split_manifest  # noqa: E402
from modules.runtime import resolve_torch_device  # noqa: E402
from modules.tracking import SortTracker  # noqa: E402
from modules.waymo_open_dataset import ToolKit, _L_BOX  # noqa: E402

CLASS_COLORS = {0: (80, 200, 255), 1: (255, 120, 255), 2: (120, 255, 120)}      # BGR, by LiDAR class id
BEV_SIZE = 720


def track_color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 47) % 180
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def text(image, label: str, origin: tuple[int, int], scale: float = 0.7, color=(255, 255, 255)) -> None:
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def banner(image, label: str, origin: tuple[int, int], scale: float = 0.7, color=(255, 255, 255)) -> None:
    """Caption on a dark strip, so it stays legible over bright sky."""
    (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = origin
    top, left = max(y - height - 8, 0), max(x - 8, 0)
    region = image[top:y + 10, left:x + width + 8]
    region[:] = (region * 0.35).astype(np.uint8)
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def build(config: str, weights: str | None, device: str):
    cfg = yaml.safe_load(Path(config).read_text())
    if weights:
        cfg["resume_weights"] = weights
    detector = build_detector(cfg)
    return cfg, detector, detector.build_model(cfg, resolve_torch_device(device)).eval()


def box_corners(cx: float, cy: float, dx: float, dy: float, heading: float) -> np.ndarray:
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]]) / 2
    rotation = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    return local @ rotation.T + np.array([cx, cy])


def render_bev(points: np.ndarray, gt: list, preds: list, view_range: float) -> np.ndarray:
    """Top-down view, ego at the centre, vehicle +x (forward) up the image."""
    scale = BEV_SIZE / (2 * view_range)
    canvas = np.full((BEV_SIZE, BEV_SIZE, 3), 18, np.uint8)

    def to_px(xy: np.ndarray) -> np.ndarray:
        return np.stack([BEV_SIZE / 2 - xy[:, 1] * scale, BEV_SIZE / 2 - xy[:, 0] * scale], axis=1)

    for ring in (10, 20, 30, 40, 50):
        if ring <= view_range:
            cv2.circle(canvas, (BEV_SIZE // 2, BEV_SIZE // 2), int(ring * scale), (45, 45, 45), 1)
            text(canvas, f"{ring} m", (BEV_SIZE // 2 + 4, BEV_SIZE // 2 - int(ring * scale) + 14), 0.4, (110, 110, 110))
    keep = (np.abs(points[:, 0]) < view_range) & (np.abs(points[:, 1]) < view_range) & (points[:, 2] > -2) & (points[:, 2] < 6)
    pts = points[keep]
    px = to_px(pts[:, :2]).astype(int)
    heights = np.clip((pts[:, 2] + 2) / 6, 0, 1)
    colors = cv2.applyColorMap((heights * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
    canvas[px[:, 1].clip(0, BEV_SIZE - 1), px[:, 0].clip(0, BEV_SIZE - 1)] = colors
    for box in gt:                                   # ground truth: green outline
        corners = to_px(box_corners(box[0], box[1], box[3], box[4], box[6])).astype(np.int32)
        cv2.polylines(canvas, [corners], True, (60, 220, 60), 2, cv2.LINE_AA)
    for item in preds:                               # LiDAR detector: orange outline
        b = item["box"]
        corners = to_px(box_corners(b[0], b[1], b[3], b[4], b[6])).astype(np.int32)
        cv2.polylines(canvas, [corners], True, (40, 150, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(canvas, (BEV_SIZE // 2, BEV_SIZE // 2), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
    return canvas


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    for code in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, size)
        if writer.isOpened():
            return writer, code
    raise RuntimeError("no usable video codec in this OpenCV build")


def suggest(args) -> None:
    """Rank 90-frame windows of the evaluation segments by tracking events and object density."""
    from track_eval import collect
    from track_replay import replay_segment

    cfg = yaml.safe_load(Path(args.camera_config).read_text())
    run_dir = Path(args.camera_run_dir)
    best = json.loads((run_dir / "tracking" / "tuning_dev.json").read_text())["best"]
    settings = {k: best[k] for k in ("score_thresh", "iou_threshold", "max_age", "min_hits")}
    class_names = cfg["model"]["class_names"]
    toolkit = ToolKit(split="validation")
    segments = load_split_manifest(cfg["data"]["manifest"])["eval"]
    frames = collect(toolkit, segments, 1, cfg, run_dir, "auto")
    rows = []
    for segment, segment_frames in frames.items():
        records, events = replay_segment(segment_frames, class_names, settings, 0.5)
        for start in range(0, max(len(records) - args.frames, 1), 10):
            window = range(start, start + args.frames)
            inside = [e for e in events if e["frame"] in window]
            objects = np.mean([sum(1 for g in records[i]["gts"] if g["label"] in (0, 1)) for i in window])
            switches = sum(1 for e in inside if e["type"] == "switch")
            assoc = sum(1 for e in inside if e["type"] == "miss" and e["cause"] == "association")
            matched = sum(1 for i in window for s in records[i]["status"].values() if s["state"] == "matched")
            rows.append({"segment": segment, "start": start, "switches": switches, "association_misses": assoc,
                         "mean_objects": round(float(objects), 1), "matched_boxes": matched,
                         "score": switches * 3 + min(assoc, 40) / 10 + matched / 50 + objects})
    rows.sort(key=lambda r: r["score"], reverse=True)
    seen, shown = set(), 0
    for row in rows:
        if row["segment"] in seen:
            continue
        seen.add(row["segment"])
        print(json.dumps(row))
        shown += 1
        if shown >= 6:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suggest", action="store_true", help="List candidate clip windows and exit.")
    parser.add_argument("--segment")
    parser.add_argument("--start", type=int, default=0, help="First frame of the clip within the segment.")
    parser.add_argument("--frames", type=int, default=90, help="Clip length in frames (10 Hz).")
    parser.add_argument("--out", type=Path, default=Path("demo.mp4"))
    parser.add_argument("--camera-config", default="configs/yolov8n_hires_100.yaml")
    parser.add_argument("--camera-run-dir", default="runs/yolov8n_hires_100-v1.2-s42",
                        help="Holds tracking/tuning_dev.json (tracker settings) and the default weights.")
    parser.add_argument("--camera-weights", default=None, help="Default: <camera-run-dir>/checkpoints/latest.pt")
    parser.add_argument("--lidar-config", default="configs/centerpoint.yaml")
    parser.add_argument("--lidar-weights", default="runs/centerpoint-v1.2-s42/checkpoints/latest.pt")
    parser.add_argument("--lidar-score", type=float, default=0.4, help="Minimum LiDAR detection score drawn.")
    parser.add_argument("--range", type=float, default=50.0, help="BEV half-width in metres.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.suggest:
        suggest(args)
        return
    if not args.segment:
        parser.error("--segment is required (use --suggest to find one)")

    run_dir = Path(args.camera_run_dir)
    best = json.loads((run_dir / "tracking" / "tuning_dev.json").read_text())["best"]
    tracker = SortTracker(best["iou_threshold"], int(best["max_age"]), int(best["min_hits"]))
    gate = best["score_thresh"]
    cam_cfg, cam_det, cam_model = build(args.camera_config, args.camera_weights or str(run_dir / "checkpoints" / "latest.pt"),
                                        args.device)
    lid_cfg, lid_det, lid_model = build(args.lidar_config, args.lidar_weights, args.device)

    toolkit = ToolKit(split="validation")
    toolkit.assign_segment(args.segment)
    timestamps = sorted(int(t) for t in toolkit.get_timestamps())
    end = args.start + args.frames
    writer = None
    for index, timestamp in enumerate(timestamps[:end]):
        image = toolkit.load_camera_frame(timestamp, 1)
        detections = [d for d in cam_det.predict(cam_model, image, cam_cfg) if d["score"] >= gate]
        tracks = tracker.update(detections)          # tracking runs from the segment start so ids are established
        if index < args.start:
            continue
        left = image.copy()
        for track in tracks:
            x1, y1, x2, y2 = (int(v) for v in track["box"])
            color = track_color(track["id"])
            cv2.rectangle(left, (x1, y1), (x2, y2), color, 4)
            text(left, f"#{track['id']}", (x1, max(y1 - 8, 30)), 1.4, color)
        left = cv2.resize(left, (int(left.shape[1] * BEV_SIZE / left.shape[0]), BEV_SIZE), interpolation=cv2.INTER_AREA)

        sets = toolkit.load_lidar_points_xyzi(timestamp, top_only=True)
        points = max(sets, key=len) if sets else np.zeros((0, 4), np.float32)
        preds = [p for p in lid_det.predict(lid_model, points, lid_cfg) if p["score"] >= args.lidar_score]
        gt = [[float(row[f"{_L_BOX}.box.{k}"]) for k in ("center.x", "center.y", "center.z", "size.x", "size.y", "size.z", "heading")]
              for _, row in toolkit.load_lidar_boxes(timestamp).iterrows()
              if int(row[f"{_L_BOX}.num_top_lidar_points_in_box"]) >= 5]
        right = render_bev(points, gt, preds, args.range)

        banner(left, "Camera: YOLOv8n 640x960 + SORT tracker", (14, 34))
        banner(right, "LiDAR (TOP), bird's-eye view", (14, 34))
        text(right, "green: ground truth", (14, BEV_SIZE - 58), 0.6, (60, 220, 60))
        text(right, f"orange: CenterPoint, score >= {args.lidar_score:g}", (14, BEV_SIZE - 32), 0.6, (40, 150, 255))
        body = np.hstack([left, right])
        bar = np.zeros((70, body.shape[1], 3), np.uint8)
        text(bar, f"Waymo Open Dataset, validation segment {args.segment}, frames {args.start}-{end - 1}. "
                  "Different models in each panel (not fusion); low-budget baselines.", (14, 28), 0.55)
        text(bar, "Made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset License Agreement "
                  "for Non-Commercial Use: waymo.com/open/terms", (14, 54), 0.5, (200, 200, 200))
        frame = np.vstack([body, bar])
        if writer is None:
            writer, codec = open_writer(args.out, 10.0, (frame.shape[1], frame.shape[0]))
            print(f"writing {args.out} ({frame.shape[1]}x{frame.shape[0]}, codec {codec})", flush=True)
        writer.write(frame)
        if (index - args.start) % 10 == 0:
            print(f"  frame {index - args.start + 1}/{args.frames}", flush=True)
    writer.release()
    print(f"done: {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
