"""HOTA: analytic cases (values worked out by hand from the TrackEval definition)."""

import math

import numpy as np
import pytest

from modules.hota import ALPHAS, combine, hota_sequence, summarize
from track_eval import score

BOX = [10, 10, 60, 60]
N = 10


def run(frames):
    return summarize(combine([hota_sequence(frames)]))


def frame(gt_ids, tracker_ids, gt_box=BOX, tracker_box=BOX):
    return (gt_ids, [gt_box] * len(gt_ids), tracker_ids, [tracker_box] * len(tracker_ids))


def test_alpha_grid_matches_trackeval():
    assert len(ALPHAS) == 19 and math.isclose(ALPHAS[0], 0.05) and math.isclose(ALPHAS[-1], 0.95)


def test_identity_scores_one():
    result = run([frame(["a"], ["a"]) for _ in range(N)])
    assert result["HOTA"] == pytest.approx(1.0) and result["DetA"] == 1.0 and result["AssA"] == 1.0
    assert result["LocA"] == pytest.approx(1.0)


def test_id_swap_halfway_halves_association_accuracy():
    frames = [frame(["a"], [1 if t < N // 2 else 2]) for t in range(N)]
    result = run(frames)
    assert result["DetA"] == pytest.approx(1.0)
    assert result["AssA"] == pytest.approx(0.5)               # each tracker id covers half of the ground-truth track
    assert result["HOTA"] == pytest.approx(math.sqrt(0.5))
    assert result["AssPr"] == pytest.approx(1.0) and result["AssRe"] == pytest.approx(0.5)


def test_missing_half_the_frames_halves_detection_accuracy():
    frames = [frame(["a"], ["x"] if t < N // 2 else []) for t in range(N)]
    result = run(frames)
    assert result["DetA"] == pytest.approx(0.5) and result["DetRe"] == pytest.approx(0.5) and result["DetPr"] == 1.0
    assert result["AssA"] == pytest.approx(0.5)               # 5 matches over a 10-frame ground-truth track
    assert result["HOTA"] == pytest.approx(0.5)


def test_no_detections_and_no_ground_truth_score_zero():
    assert run([frame(["a"], []) for _ in range(N)])["HOTA"] == 0.0
    assert run([frame([], ["x"]) for _ in range(N)])["HOTA"] == 0.0


def test_localisation_limits_which_alphas_match():
    shifted = [40, 10, 90, 60]                                  # IoU with BOX is exactly 0.25
    result = run([frame(["a"], ["a"], tracker_box=shifted) for _ in range(N)])
    matched = int(np.sum(ALPHAS <= 0.25 + 1e-9))                # alphas 0.05 .. 0.25 match, the rest do not
    assert matched == 5
    assert result["HOTA"] == pytest.approx(matched / len(ALPHAS))
    assert result["DetA"] == pytest.approx(matched / len(ALPHAS))


def test_false_positive_tracks_lower_detection_precision_only():
    frames = [frame(["a"], ["a", "ghost"], gt_box=BOX) for _ in range(N)]
    frames = [(g, gb, t, [BOX, [200, 200, 240, 240]]) for g, gb, t, _ in frames]
    result = run(frames)
    assert result["DetRe"] == pytest.approx(1.0) and result["DetPr"] == pytest.approx(0.5)
    assert result["DetA"] == pytest.approx(0.5)


def test_sequences_are_combined_by_summing_counts():
    perfect = hota_sequence([frame(["a"], ["a"]) for _ in range(N)])
    empty = hota_sequence([frame(["b"], []) for _ in range(N)])
    result = summarize(combine([perfect, empty]))
    assert result["DetA"] == pytest.approx(0.5)               # 10 TP, 10 FN
    assert result["AssA"] == pytest.approx(1.0)               # association is weighted by TP, so the empty sequence adds none


def test_score_reports_hota_next_to_mota_and_idf1():
    classes = ["vehicle", "pedestrian", "cyclist"]
    seq = []
    for _ in range(6):
        gts = [{"id": "car", "label": 0, "box": [10, 10, 60, 60]}, {"id": "ped", "label": 1, "box": [200, 100, 220, 160]}]
        seq.append((gts, [{"box": g["box"], "label": g["label"], "score": 0.9} for g in gts]))
    for kwargs in ({"identity_oracle": True}, {"min_hits": 1}):
        report = score({"seg": seq}, classes, **kwargs)
        assert report["weighted_hota"] == pytest.approx(1.0)
        assert report["weighted_deta"] == pytest.approx(1.0) and report["weighted_assa"] == pytest.approx(1.0)
        assert report["per_class"]["vehicle"]["hota"] == pytest.approx(1.0)
        assert report["weighted_mota"] == 1.0              # existing metrics are unchanged
