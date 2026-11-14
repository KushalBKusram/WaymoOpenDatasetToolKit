"""SORT-style 2-D multi-object tracker for tracking-by-detection baselines.

Each track keeps a constant-velocity Kalman filter over (cx, cy, w, h); detections are
associated to predicted boxes of the same class by Hungarian matching on IoU.

Optional appearance ReID: when detections carry an L2-normalised 'embedding', each track keeps an
exponential moving average of its embeddings. Association can blend cosine distance into the IoU cost
(`reid_weight`), and a second stage re-links tracks that were lost for a few frames to unmatched
detections of the same class with similar appearance near where the track should be (`reid_gate`).
With `reid_weight=0` and `reid_gate=0` (the defaults) the tracker is the plain IoU tracker.

Optional LiDAR 3-D gating: when detections carry a world-frame position 'xyz_world' (see `modules.lidar_gate`), each
track keeps its last world position and a ground-plane velocity. `lidar_gate` rejects a track-detection pair whose
ground-plane distance to the track's predicted position exceeds `lidar_gate * sqrt(frames since the track's last 3-D
update)` metres, and `lidar_weight` blends the normalised distance into the association cost. Pairs where either side
has no 3-D estimate are not affected. With `lidar_gate=0` and `lidar_weight=0` (the defaults) nothing changes.
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


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance (1 - similarity) between rows of two L2-normalised matrices, in [0, 2]."""
    return np.clip(1.0 - np.asarray(a, dtype=float) @ np.asarray(b, dtype=float).T, 0.0, 2.0)


def ema_embedding(old: np.ndarray, new: np.ndarray, ema: float) -> np.ndarray:
    """Exponential moving average of two embeddings, re-normalised to unit length."""
    mixed = ema * np.asarray(old, dtype=float) + (1.0 - ema) * np.asarray(new, dtype=float)
    norm = np.linalg.norm(mixed)
    return mixed / norm if norm > 0 else mixed


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

    def __init__(self, track_id: int, box: np.ndarray, label: int, score: float, embedding: np.ndarray | None = None,
                 xyz: np.ndarray | None = None, frame: int = 0):
        self.id, self.label, self.score = track_id, label, score
        self.embedding = None if embedding is None else np.asarray(embedding, dtype=float)
        self.x = np.concatenate([_to_state(box), np.zeros(4)])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e3, 1e3, 1e3, 1e3])
        self.hits, self.age, self.since_update = 1, 0, 0
        self.pos3 = None if xyz is None else np.asarray(xyz, dtype=float)     # last world-frame 3-D position
        self.vel3, self.last3 = np.zeros(3), (frame if xyz is not None else None)

    def predict(self) -> np.ndarray:
        process = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001, 0.0001])
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + process
        self.age += 1
        self.since_update += 1
        return _to_box(self.x)

    def predicted_position(self, frame: int) -> np.ndarray | None:
        """Last 3-D position moved on by the ground-plane velocity, or None if the track has none."""
        if self.pos3 is None:
            return None
        return self.pos3 + self.vel3 * (frame - self.last3)

    def update(self, box: np.ndarray, score: float, embedding: np.ndarray | None = None, ema: float = 0.9,
               xyz: np.ndarray | None = None, frame: int = 0) -> None:
        if xyz is not None:
            xyz = np.asarray(xyz, dtype=float)
            if self.pos3 is not None and 1 <= frame - self.last3 <= 10:
                velocity = np.append((xyz[:2] - self.pos3[:2]) / (frame - self.last3), 0.0)
                speed = float(np.linalg.norm(velocity))
                if speed > 3.0:                                       # at most 3 m per 0.1 s frame
                    velocity *= 3.0 / speed
                self.vel3 = 0.5 * self.vel3 + 0.5 * velocity
            else:
                self.vel3 = np.zeros(3)
            self.pos3, self.last3 = xyz, frame
        if embedding is not None:
            self.embedding = (np.asarray(embedding, dtype=float) if self.embedding is None
                              else ema_embedding(self.embedding, embedding, ema))
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
        reid_weight: weight w of the cosine distance in the association cost,
            cost = (1 - w) * (1 - IoU) + w * cosine_distance (still gated by `iou_threshold`).
        reid_gate: 0 disables the re-link stage. Otherwise unmatched tracks are re-linked to unmatched
            detections of the same class when the detection centre is within
            `reid_gate * track box diagonal * sqrt(frames lost)` of the track's predicted centre.
        reid_thresh: maximum cosine distance for a re-link.
        ema: weight of the old embedding in a track's moving average.
        lidar_gate: 0 disables LiDAR gating; otherwise the base distance in metres (see the module docstring).
        lidar_weight: weight of the normalised 3-D distance in the association cost.
        lidar_min_points: detections whose 3-D estimate used fewer LiDAR points are treated as having none.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5, min_hits: int = 2, *,
                 reid_weight: float = 0.0, reid_gate: float = 0.0, reid_thresh: float = 0.4, ema: float = 0.9,
                 lidar_gate: float = 0.0, lidar_weight: float = 0.0, lidar_min_points: int = 0):
        self.iou_threshold, self.max_age, self.min_hits = iou_threshold, max_age, min_hits
        self.reid_weight, self.reid_gate, self.reid_thresh, self.ema = reid_weight, reid_gate, reid_thresh, ema
        self.lidar_gate, self.lidar_weight, self.lidar_min_points = lidar_gate, lidar_weight, lidar_min_points
        self.tracks: list[_Track] = []
        self._next_id = 1
        self._frame = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """detections: [{'box': xyxy, 'label': int, 'score': float, 'embedding'?: unit vector}] -> [{'id', 'box', 'label', 'score'}]."""
        self._frame += 1
        predicted = np.array([track.predict() for track in self.tracks]).reshape(-1, 4)
        boxes = np.array([d["box"] for d in detections], dtype=float).reshape(-1, 4)
        labels = np.array([d["label"] for d in detections], dtype=int)
        track_labels = np.array([t.label for t in self.tracks], dtype=int)
        det_emb = [d.get("embedding") for d in detections]
        use_reid = (self.reid_weight > 0 or self.reid_gate > 0) and any(e is not None for e in det_emb)
        det_xyz = [self._usable_xyz(d) for d in detections]
        use_lidar = (self.lidar_gate > 0 or self.lidar_weight > 0) and any(x is not None for x in det_xyz)
        lidar_dist = lidar_thr = None

        base = 1.0 - iou_matrix(predicted, boxes)
        cost = base
        if base.size and use_reid and self.reid_weight > 0:
            cos = np.full(base.shape, 0.5)               # neutral distance when an embedding is missing
            for t, track in enumerate(self.tracks):
                if track.embedding is None:
                    continue
                for d, emb in enumerate(det_emb):
                    if emb is not None:
                        cos[t, d] = cosine_distance(track.embedding[None], np.asarray(emb)[None])[0, 0]
            cost = (1.0 - self.reid_weight) * base + self.reid_weight * cos
        if base.size and use_lidar:
            lidar_dist, lidar_thr = self._lidar_distances(det_xyz)
            if self.lidar_weight > 0:
                norm = np.where(np.isnan(lidar_dist), 0.5, np.clip(lidar_dist / lidar_thr, 0.0, 1.5))
                cost = (1.0 - self.lidar_weight) * cost + self.lidar_weight * norm
        if cost.size:
            cost = cost.copy() if cost is base else cost
            cost[track_labels[:, None] != labels[None, :]] = 1e6     # never match across classes
            cost[base > 1.0 - self.iou_threshold] = 1e6
            if lidar_dist is not None and self.lidar_gate > 0:
                cost[lidar_dist > lidar_thr] = 1e6                   # NaN (no estimate) compares False: not gated
        matched_tracks, matched_dets = set(), set()
        if cost.size:
            for t, d in zip(*linear_sum_assignment(cost)):
                if cost[t, d] < 1e6:
                    self.tracks[t].update(boxes[d], float(detections[d]["score"]), det_emb[d], self.ema,
                                          det_xyz[d], self._frame)
                    matched_tracks.add(t)
                    matched_dets.add(d)
        if use_reid and self.reid_gate > 0 and len(self.tracks):
            self._relink(predicted, boxes, labels, det_emb, detections, matched_tracks, matched_dets,
                         det_xyz, lidar_dist, lidar_thr)
        for d in range(len(detections)):
            if d not in matched_dets:
                self.tracks.append(_Track(self._next_id, boxes[d], int(labels[d]),
                                          float(detections[d]["score"]), det_emb[d], det_xyz[d], self._frame))
                self._next_id += 1
        self.tracks = [t for t in self.tracks if t.since_update <= self.max_age]
        return [
            {"id": t.id, "box": [float(v) for v in t.box], "label": t.label, "score": t.score}
            for t in self.tracks
            if t.since_update == 0 and (t.hits >= self.min_hits or self._frame <= self.min_hits)
        ]

    def _usable_xyz(self, detection: dict):
        xyz = detection.get("xyz_world")
        if xyz is None or detection.get("n3d", 1 << 30) < self.lidar_min_points:
            return None
        return np.asarray(xyz, dtype=float)

    def _lidar_distances(self, det_xyz):
        """(tracks x detections) ground-plane distance to each track's predicted position and the gate radius.

        NaN where the track or the detection has no 3-D estimate. The radius grows with the square root of the
        frames since the track's last 3-D update; with `lidar_gate=0` a 5 m scale only normalises the blended cost.
        """
        base_gate = self.lidar_gate if self.lidar_gate > 0 else 5.0
        dist = np.full((len(self.tracks), len(det_xyz)), np.nan)
        thr = np.ones_like(dist)
        for t, track in enumerate(self.tracks):
            position = track.predicted_position(self._frame)
            if position is None:
                continue
            thr[t, :] = base_gate * np.sqrt(max(self._frame - track.last3, 1))
            for d, xyz in enumerate(det_xyz):
                if xyz is not None:
                    dist[t, d] = float(np.linalg.norm(position[:2] - xyz[:2]))
        return dist, thr

    def _relink(self, predicted, boxes, labels, det_emb, detections, matched_tracks, matched_dets,
                det_xyz=None, lidar_dist=None, lidar_thr=None) -> None:
        """Stage 2: re-link lost tracks to unmatched detections by appearance inside a spatial gate."""
        lost = [t for t in range(len(self.tracks)) if t not in matched_tracks and self.tracks[t].embedding is not None]
        free = [d for d in range(len(detections)) if d not in matched_dets and det_emb[d] is not None]
        if not lost or not free:
            return
        cost = np.full((len(lost), len(free)), 1e6)
        for i, t in enumerate(lost):
            track = self.tracks[t]
            box = predicted[t]
            diagonal = float(np.hypot(box[2] - box[0], box[3] - box[1]))
            centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            radius = self.reid_gate * diagonal * np.sqrt(max(track.since_update, 1))
            for j, d in enumerate(free):
                if labels[d] != track.label:
                    continue
                det_centre = np.array([(boxes[d][0] + boxes[d][2]) / 2, (boxes[d][1] + boxes[d][3]) / 2])
                if np.linalg.norm(det_centre - centre) > radius:
                    continue
                if (lidar_dist is not None and self.lidar_gate > 0 and not np.isnan(lidar_dist[t, d])
                        and lidar_dist[t, d] > lidar_thr[t, d]):
                    continue
                distance = cosine_distance(track.embedding[None], np.asarray(det_emb[d])[None])[0, 0]
                if distance < self.reid_thresh:
                    cost[i, j] = distance
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] < 1e6:
                t, d = lost[i], free[j]
                self.tracks[t].update(boxes[d], float(detections[d]["score"]), det_emb[d], self.ema,
                                      None if det_xyz is None else det_xyz[d], self._frame)
                matched_tracks.add(t)
                matched_dets.add(d)
