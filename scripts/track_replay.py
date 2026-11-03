"""Replay camera tracking failures: ID switches, detector misses and association misses.

Re-runs the tracker on cached detections (see scripts/track_eval.py), matches tracks to
ground truth frame by frame, and attributes every missed ground-truth box to its cause:

  detector     no detection overlaps the box at all (IoU >= match_iou)
  gate         a detection overlaps it but scores below the tracker's score gate
  association  a detection above the gate overlaps it, yet no track covers it
               (track not yet confirmed, dropped, or taken by another object)

It writes a JSON summary and image strips around the worst events.

  python scripts/track_replay.py --config configs/yolov8n_hires_100.yaml \
      --run-dir runs/yolov8n_hires_100-v1.2-s42
"""

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from modules.benchmark import load_split_manifest  # noqa: E402
from modules.tracking import SortTracker, iou_matrix  # noqa: E402
from modules.waymo_open_dataset import ToolKit  # noqa: E402
from track_eval import collect  # noqa: E402

CONTINUITY_BONUS = 0.5      # prefer keeping the previous gt<->track pairing, as MOT scoring does


def match(gts: list[dict], tracks: list[dict], previous: dict, min_iou: float) -> dict[int, int]:
    """gt index -> track index for one class and frame (IoU >= min_iou, Hungarian, sticky)."""
    if not gts or not tracks:
        return {}
    overlap = iou_matrix(np.array([g["box"] for g in gts], float), np.array([t["box"] for t in tracks], float))
    cost = np.where(overlap >= min_iou, 1.0 - overlap, 1e6)
    for gi, g in enumerate(gts):
        for ti, t in enumerate(tracks):
            if previous.get(g["id"]) == t["id"] and cost[gi, ti] < 1e6:
                cost[gi, ti] -= CONTINUITY_BONUS
    return {int(g): int(t) for g, t in zip(*linear_sum_assignment(cost)) if cost[g, t] < 1e6}


def replay_segment(frames: list, class_names: list[str], settings: dict, match_iou: float) -> tuple[list, dict]:
    """Per-frame records plus event counts for one segment."""
    tracker = SortTracker(settings["iou_threshold"], settings["max_age"], settings["min_hits"])
    last_track: dict[str, int] = {}
    last_seen: dict[str, int] = {}
    was_matched: dict[str, bool] = {}
    records, events = [], []
    for frame_index, (gts, dets) in enumerate(frames):
        gated = [d for d in dets if d["score"] >= settings["score_thresh"]]
        tracks = tracker.update(gated)
        record = {"frame": frame_index, "gts": gts, "dets": gated, "tracks": tracks, "status": {}}
        for label, name in enumerate(class_names):
            g = [x for x in gts if x["label"] == label]
            t = [x for x in tracks if x["label"] == label]
            pairs = match(g, t, last_track, match_iou)
            for gi, gt in enumerate(g):
                gid = gt["id"]
                if gi in pairs:
                    track_id = t[pairs[gi]]["id"]
                    status = {"state": "matched", "track": track_id, "class": name}
                    if gid in last_track and last_track[gid] != track_id:
                        gap = frame_index - last_seen[gid]
                        events.append({"type": "switch", "frame": frame_index, "gt": gid, "class": name,
                                       "from": last_track[gid], "to": track_id, "gap": gap, "box": gt["box"]})
                    if was_matched.get(gid) is False:
                        events.append({"type": "fragment", "frame": frame_index, "gt": gid, "class": name})
                    last_track[gid], last_seen[gid], was_matched[gid] = track_id, frame_index, True
                else:
                    same = [d for d in dets if d["label"] == label]
                    ious = iou_matrix(np.array([gt["box"]], float), np.array([d["box"] for d in same], float).reshape(-1, 4))[0] \
                        if same else np.zeros(0)
                    over = [d for d, v in zip(same, ious) if v >= match_iou]
                    if not over:
                        cause = "detector"
                    elif all(d["score"] < settings["score_thresh"] for d in over):
                        cause = "gate"
                    else:
                        cause = "association"
                    status = {"state": "missed", "cause": cause, "class": name}
                    events.append({"type": "miss", "frame": frame_index, "gt": gid, "class": name,
                                   "cause": cause, "box": gt["box"]})
                    if gid in was_matched:
                        was_matched[gid] = False
                record["status"][gid] = status
        records.append(record)
    return records, events


def short_tracks(records: list, minimum: int) -> int:
    lengths = Counter(t["id"] for r in records for t in r["tracks"])
    return sum(1 for v in lengths.values() if v < minimum)


def draw(image: np.ndarray, record: dict, label: str) -> np.ndarray:
    image = image.copy()
    for det in record["dets"]:
        x1, y1, x2, y2 = (int(v) for v in det["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 1)
    for gt in record["gts"]:
        x1, y1, x2, y2 = (int(v) for v in gt["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
    for track in record["tracks"]:
        x1, y1, x2, y2 = (int(v) for v in track["box"])
        hue = (track["id"] * 47) % 180
        color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(image, f"#{track['id']}", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(image, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(image, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
    return image


def strip(toolkit: ToolKit, camera: int, timestamps: list, records: list, event: dict, span: int, out: Path) -> None:
    """Crop around the event's box over frames [t-span, t+span] and write one image row."""
    x1, y1, x2, y2 = event["box"]
    cx, cy, size = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1, 160) * 2.2
    tiles = []
    for index in range(max(event["frame"] - span, 0), min(event["frame"] + span, len(records) - 1) + 1):
        image = draw(toolkit.load_camera_frame(timestamps[index], camera), records[index],
                     f"t{index - event['frame']:+d}")
        h, w = image.shape[:2]
        left, top = int(max(min(cx - size / 2, w - size), 0)), int(max(min(cy - size / 2, h - size), 0))
        crop = image[top:top + int(size), left:left + int(size)]
        tiles.append(cv2.resize(crop, (320, 320)))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack(tiles))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--examples", type=int, default=4, help="Image strips per event type.")
    parser.add_argument("--span", type=int, default=3, help="Frames before and after each event in a strip.")
    parser.add_argument("--short-track", type=int, default=5, help="Tracks shorter than this many frames count as short.")
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(args.run_dir)
    class_names = cfg["model"].get("class_names", ["vehicle", "pedestrian", "cyclist", "sign"])
    tuning = json.loads((run_dir / "tracking" / "tuning_dev.json").read_text())["best"]
    settings = {k: tuning[k] for k in ("score_thresh", "iou_threshold", "max_age", "min_hits")}

    toolkit = ToolKit(split="validation")
    manifest = cfg["data"].get("manifest")
    segments = load_split_manifest(manifest)["eval"] if manifest else toolkit.list_segments()
    segments = segments[:args.segments or cfg.get("eval", {}).get("segs", 10)]
    frames = collect(toolkit, segments, args.camera, cfg, run_dir, args.device)

    all_events, per_segment, totals = [], {}, Counter()
    gt_frames: Counter = Counter()
    kept: dict[str, tuple[list, list]] = {}
    for segment, segment_frames in frames.items():
        records, events = replay_segment(segment_frames, class_names, settings, args.match_iou)
        toolkit.assign_segment(segment)
        timestamps = sorted(int(t) for t in toolkit.get_timestamps())
        for e in events:
            e["segment"] = segment
        all_events += events
        kept[segment] = (records, timestamps)
        per_segment[segment] = {"frames": len(records), "short_tracks": short_tracks(records, args.short_track)}
        for r in records:
            for status in r["status"].values():
                gt_frames[status["class"]] += 1

    summary = {"run": run_dir.name, "settings": settings, "segments": segments, "gt_boxes": dict(gt_frames),
               "per_class": {}, "short_tracks": sum(v["short_tracks"] for v in per_segment.values())}
    for name in class_names:
        mine = [e for e in all_events if e["class"] == name]
        if not gt_frames[name]:
            continue
        misses = Counter(e["cause"] for e in mine if e["type"] == "miss")
        switches = [e for e in mine if e["type"] == "switch"]
        side = lambda e: float(np.sqrt((e["box"][2] - e["box"][0]) * (e["box"][3] - e["box"][1])))  # noqa: E731
        by_cause = {c: [side(e) for e in mine if e["type"] == "miss" and e["cause"] == c] for c in ("detector", "association")}
        summary["per_class"][name] = {
            "median_miss_box_side_px": {c: round(float(np.median(v)), 1) if v else None for c, v in by_cause.items()},
            "gt_boxes": gt_frames[name], "missed": sum(misses.values()),
            "missed_by_cause": {c: misses.get(c, 0) for c in ("detector", "gate", "association")},
            "id_switches": len(switches),
            "switches_after_gap": sum(1 for e in switches if e["gap"] > 1),
            "fragmentations": sum(1 for e in mine if e["type"] == "fragment")}

    out_dir = run_dir / "tracking" / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "failure_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    if args.no_images:
        return
    area = lambda e: (e["box"][2] - e["box"][0]) * (e["box"][3] - e["box"][1])  # noqa: E731
    picks = {
        "switch": sorted((e for e in all_events if e["type"] == "switch" and e["gap"] <= 1), key=area, reverse=True),
        "switch_after_gap": sorted((e for e in all_events if e["type"] == "switch" and e["gap"] > 1), key=area, reverse=True),
        "miss_association": sorted((e for e in all_events if e["type"] == "miss" and e["cause"] == "association"
                                    and e["class"] == "vehicle"), key=area, reverse=True),
        "miss_detector": sorted((e for e in all_events if e["type"] == "miss" and e["cause"] == "detector"
                                 and e["class"] == "vehicle"), key=area, reverse=True),
    }
    for kind, events in picks.items():
        seen_segments: defaultdict = defaultdict(list)
        written = 0
        for event in events:
            nearby = any(abs(event["frame"] - f) < 15 for f in seen_segments[event["segment"]])
            if written >= args.examples or len(seen_segments[event["segment"]]) >= 2 or nearby:
                continue
            records, timestamps = kept[event["segment"]]
            toolkit.assign_segment(event["segment"])
            strip(toolkit, args.camera, timestamps, records, event, args.span,
                  out_dir / f"{kind}_{written + 1}_{event['segment'][:12]}_f{event['frame']}.png")
            seen_segments[event["segment"]].append(event["frame"])
            written += 1
        print(f"{kind}: {written} strips")


if __name__ == "__main__":
    main()
