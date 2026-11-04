import math

import numpy as np

from track_eval import match_distances, score

CLASSES = ["vehicle", "pedestrian", "cyclist"]


def gt(obj_id, box, label=0):
    return {"id": obj_id, "label": label, "box": list(box)}


def det(box, label=0, score=0.9):
    return {"box": list(box), "label": label, "score": score}


def static_scene(frames=6):
    """One vehicle and one pedestrian that never move, with perfect detections."""
    seq = []
    for _ in range(frames):
        gts = [gt("car", (10, 10, 60, 60), 0), gt("ped", (200, 100, 220, 160), 1)]
        seq.append((gts, [det(g["box"], g["label"]) for g in gts]))
    return {"seg": seq}


def test_identity_oracle_scores_perfectly():
    report = score(static_scene(), CLASSES, identity_oracle=True)
    assert report["weighted_mota"] == 1.0
    assert report["weighted_idf1"] == 1.0
    assert "cyclist" not in report["per_class"]                  # no ground-truth cyclists


def test_perfect_detections_through_the_tracker():
    report = score(static_scene(), CLASSES, min_hits=1)
    assert report["weighted_idf1"] == 1.0
    assert report["per_class"]["vehicle"]["num_switches"] == 0


def test_no_detections_scores_zero_recall():
    scene = static_scene()
    empty = {"seg": [(gts, []) for gts, _ in scene["seg"]]}
    report = score(empty, CLASSES)
    assert report["per_class"]["vehicle"]["recall"] == 0.0
    assert report["weighted_idf1"] == 0.0


def test_match_distances_use_one_minus_iou_and_nan_below_threshold():
    gts = [gt("a", (0, 0, 10, 10))]
    tracks = [{"box": [0, 0, 10, 10]}, {"box": [100, 100, 110, 110]}]
    dist = match_distances(gts, tracks, 0.5)
    assert dist.shape == (1, 2)
    assert dist[0, 0] == 0.0
    assert math.isnan(dist[0, 1])
    assert np.isnan(match_distances(gts, [{"box": [5, 0, 15, 10]}], 0.9)[0, 0])
