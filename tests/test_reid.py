"""Appearance ReID in the tracker: helpers, backward compatibility and the re-link behaviour."""

import numpy as np
import pytest

from modules.tracking import SortTracker, _Track, cosine_distance, ema_embedding, iou_matrix
from scipy.optimize import linear_sum_assignment


def unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


class ReferenceTracker:
    """The plain IoU tracker exactly as it was before ReID was added (association and track management)."""

    def __init__(self, iou_threshold=0.3, max_age=5, min_hits=2):
        self.iou_threshold, self.max_age, self.min_hits = iou_threshold, max_age, min_hits
        self.tracks, self._next_id, self._frame = [], 1, 0

    def update(self, detections):
        self._frame += 1
        predicted = np.array([track.predict() for track in self.tracks]).reshape(-1, 4)
        boxes = np.array([d["box"] for d in detections], dtype=float).reshape(-1, 4)
        labels = np.array([d["label"] for d in detections], dtype=int)
        track_labels = np.array([t.label for t in self.tracks], dtype=int)
        cost = 1.0 - iou_matrix(predicted, boxes)
        if cost.size:
            cost[track_labels[:, None] != labels[None, :]] = 1e6
            cost[cost > 1.0 - self.iou_threshold] = 1e6
        matched = set()
        if cost.size:
            for t, d in zip(*linear_sum_assignment(cost)):
                if cost[t, d] < 1e6:
                    self.tracks[t].update(boxes[d], float(detections[d]["score"]))
                    matched.add(d)
        for d in range(len(detections)):
            if d not in matched:
                self.tracks.append(_Track(self._next_id, boxes[d], int(labels[d]), float(detections[d]["score"])))
                self._next_id += 1
        self.tracks = [t for t in self.tracks if t.since_update <= self.max_age]
        return [{"id": t.id, "box": [float(v) for v in t.box], "label": t.label, "score": t.score}
                for t in self.tracks if t.since_update == 0 and (t.hits >= self.min_hits or self._frame <= self.min_hits)]


def random_sequence(seed: int, frames: int = 40):
    rng = np.random.default_rng(seed)
    objects = [{"c": rng.uniform(100, 900, 2), "v": rng.normal(0, 4, 2), "wh": rng.uniform(30, 90, 2),
                "label": int(rng.integers(0, 3))} for _ in range(8)]
    for _ in range(frames):
        dets = []
        for obj in objects:
            obj["c"] = obj["c"] + obj["v"]
            if rng.random() < 0.85:                                          # some missed detections
                c, wh = obj["c"] + rng.normal(0, 1.5, 2), obj["wh"] + rng.normal(0, 1, 2)
                dets.append({"box": [c[0] - wh[0] / 2, c[1] - wh[1] / 2, c[0] + wh[0] / 2, c[1] + wh[1] / 2],
                             "label": obj["label"], "score": float(rng.uniform(0.3, 0.95))})
        yield dets


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_embeddings_matches_the_plain_tracker(seed):
    new, old = SortTracker(0.3, 5, 2), ReferenceTracker(0.3, 5, 2)
    for dets in random_sequence(seed):
        assert new.update(dets) == old.update(dets)


@pytest.mark.parametrize("seed", [0, 1])
def test_reid_settings_without_embeddings_change_nothing(seed):
    new, old = SortTracker(0.3, 5, 2, reid_weight=0.5, reid_gate=2.0), ReferenceTracker(0.3, 5, 2)
    for dets in random_sequence(seed):
        assert new.update(dets) == old.update(dets)


def test_cosine_distance_and_ema():
    a, b = unit([1, 0]), unit([0, 1])
    np.testing.assert_allclose(cosine_distance(a[None], a[None]), [[0.0]], atol=1e-12)
    np.testing.assert_allclose(cosine_distance(a[None], b[None]), [[1.0]], atol=1e-12)
    np.testing.assert_allclose(cosine_distance(a[None], (-a)[None]), [[2.0]], atol=1e-12)
    mixed = ema_embedding(a, b, 0.9)
    assert abs(np.linalg.norm(mixed) - 1.0) < 1e-12 and mixed[0] > mixed[1] > 0
    np.testing.assert_allclose(ema_embedding(a, b, 1.0), a)             # ema=1 keeps the old embedding
    np.testing.assert_allclose(ema_embedding(a, b, 0.0), b)             # ema=0 takes the new one


def box_at(x: float, y: float = 200.0, size: float = 60.0) -> list[float]:
    return [x, y, x + size, y + size]


def test_relink_after_the_track_was_dropped_by_iou():
    """Object returns shifted so IoU is zero: only the appearance re-link recovers its id."""
    emb_a, emb_b = unit([1, 0, 0]), unit([0, 1, 0])

    def run(tracker):
        first = None
        for _ in range(4):
            out = tracker.update([{"box": box_at(100), "label": 0, "score": 0.9, "embedding": emb_a}])
            first = out[0]["id"]
        for _ in range(3):
            tracker.update([])
        out = tracker.update([{"box": box_at(230), "label": 0, "score": 0.9, "embedding": emb_a}])   # 130 px away, no overlap
        return first, out[0]["id"] if out else None

    first, after = run(SortTracker(0.3, 20, 1))
    assert after != first                                                 # IoU alone cannot bridge the jump
    first, after = run(SortTracker(0.3, 20, 1, reid_gate=2.0, reid_thresh=0.3))
    assert after == first

    def wrong_appearance(tracker):
        for _ in range(4):
            first = tracker.update([{"box": box_at(100), "label": 0, "score": 0.9, "embedding": emb_a}])[0]["id"]
        for _ in range(3):
            tracker.update([])
        out = tracker.update([{"box": box_at(230), "label": 0, "score": 0.9, "embedding": emb_b}])
        return first, out[0]["id"]

    first, after = wrong_appearance(SortTracker(0.3, 20, 1, reid_gate=2.0, reid_thresh=0.3))
    assert after != first                                                 # a different-looking object is not re-linked


def test_relink_respects_class_and_spatial_gate():
    emb = unit([1, 0, 0])
    tracker = SortTracker(0.3, 20, 1, reid_gate=1.0, reid_thresh=0.3)
    for _ in range(4):
        first = tracker.update([{"box": box_at(100), "label": 0, "score": 0.9, "embedding": emb}])[0]["id"]
    tracker.update([])
    far = tracker.update([{"box": box_at(900), "label": 0, "score": 0.9, "embedding": emb}])       # outside the gate
    assert far[0]["id"] != first
    tracker = SortTracker(0.3, 20, 1, reid_gate=2.0, reid_thresh=0.3)
    for _ in range(4):
        first = tracker.update([{"box": box_at(100), "label": 0, "score": 0.9, "embedding": emb}])[0]["id"]
    tracker.update([])
    other_class = tracker.update([{"box": box_at(130), "label": 1, "score": 0.9, "embedding": emb}])
    assert other_class[0]["id"] != first


def test_blended_cost_decides_by_appearance_when_iou_is_tied():
    emb_a, emb_b = unit([1, 0, 0]), unit([0, 1, 0])
    tracker = SortTracker(0.1, 10, 1, reid_weight=0.8)
    for _ in range(3):
        out = tracker.update([{"box": box_at(100), "label": 0, "score": 0.9, "embedding": emb_a},
                              {"box": box_at(130), "label": 0, "score": 0.9, "embedding": emb_b}])
    id_a, id_b = [t["id"] for t in sorted(out, key=lambda t: t["box"][0])]
    # Two detections at the same box: both overlap each track equally, so IoU cannot tell them apart.
    tracker.update([{"box": box_at(115), "label": 0, "score": 0.9, "embedding": emb_b},
                    {"box": box_at(115), "label": 0, "score": 0.9, "embedding": emb_a}])
    by_id = {track.id: track.embedding for track in tracker.tracks}
    assert float(by_id[id_a] @ emb_a) > float(by_id[id_a] @ emb_b)
    assert float(by_id[id_b] @ emb_b) > float(by_id[id_b] @ emb_a)
