"""SORT-style 2-D multi-object tracker for tracking-by-detection baselines.

Each track keeps a constant-velocity Kalman filter over (cx, cy, w, h); detections are
associated to predicted boxes of the same class by Hungarian matching on IoU.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _to_state(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]])


def _to_box(state: np.ndarray) -> np.ndarray:
    cx, cy, w, h = state[:4]
    w, h = max(w, 1.0), max(h, 1.0)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class _Track:
    """Kalman filter over [cx, cy, w, h, vx, vy, vw, vh] with a one-frame time step."""

    _F = np.eye(8) + np.eye(8, k=4)
    _H = np.eye(4, 8)

    def __init__(self, track_id: int, box: np.ndarray, label: int, score: float):
        self.id, self.label, self.score = track_id, label, score
        self.x = np.concatenate([_to_state(box), np.zeros(4)])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e3, 1e3, 1e3, 1e3])
        self.hits, self.age, self.since_update = 1, 0, 0

    def predict(self) -> np.ndarray:
        process = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001, 0.0001])
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + process
        self.age += 1
        self.since_update += 1
        return _to_box(self.x)

    def update(self, box: np.ndarray, score: float) -> None:
        noise = np.diag([1.0, 1.0, 10.0, 10.0])
        gain = self.P @ self._H.T @ np.linalg.inv(self._H @ self.P @ self._H.T + noise)
        self.x = self.x + gain @ (_to_state(box) - self._H @ self.x)
        self.P = (np.eye(8) - gain @ self._H) @ self.P
        self.score, self.hits, self.since_update = score, self.hits + 1, 0

    @property
    def box(self) -> np.ndarray:
        return _to_box(self.x)


class SortTracker:
    """Online tracker. Call `update(detections)` once per frame in time order.

    Args:
        iou_threshold: minimum IoU to associate a detection with a track.
        max_age: frames a track survives without a matched detection.
        min_hits: matched frames before a new track is reported (tracks are
            always reported on their creation frame for the first `min_hits` frames
            of the sequence, as in SORT).
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5, min_hits: int = 2):
        self.iou_threshold, self.max_age, self.min_hits = iou_threshold, max_age, min_hits
        self.tracks: list[_Track] = []
        self._next_id = 1
        self._frame = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """detections: [{'box': xyxy, 'label': int, 'score': float}] -> [{'id', 'box', 'label', 'score'}]."""
        self._frame += 1
        predicted = np.array([track.predict() for track in self.tracks]).reshape(-1, 4)
        boxes = np.array([d["box"] for d in detections], dtype=float).reshape(-1, 4)
        labels = np.array([d["label"] for d in detections], dtype=int)
        track_labels = np.array([t.label for t in self.tracks], dtype=int)

        cost = 1.0 - iou_matrix(predicted, boxes)
        if cost.size:
            cost[track_labels[:, None] != labels[None, :]] = 1e6     # never match across classes
            cost[cost > 1.0 - self.iou_threshold] = 1e6
        matched_tracks, matched_dets = set(), set()
        if cost.size:
            for t, d in zip(*linear_sum_assignment(cost)):
                if cost[t, d] < 1e6:
                    self.tracks[t].update(boxes[d], float(detections[d]["score"]))
                    matched_tracks.add(t)
                    matched_dets.add(d)
        for d in range(len(detections)):
            if d not in matched_dets:
                self.tracks.append(_Track(self._next_id, boxes[d], int(labels[d]),
                                          float(detections[d]["score"])))
                self._next_id += 1
        self.tracks = [t for t in self.tracks if t.since_update <= self.max_age]
        return [
            {"id": t.id, "box": [float(v) for v in t.box], "label": t.label, "score": t.score}
            for t in self.tracks
            if t.since_update == 0 and (t.hits >= self.min_hits or self._frame <= self.min_hits)
        ]
