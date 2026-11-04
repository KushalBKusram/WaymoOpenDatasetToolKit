import numpy as np

from modules.tracking import SortTracker, iou_matrix


def det(box, label=0, score=0.9):
    return {"box": list(box), "label": label, "score": score}


def test_iou_matrix_known_values():
    a = np.array([[0, 0, 2, 2]], dtype=float)
    b = np.array([[0, 0, 2, 2], [1, 0, 3, 2], [5, 5, 6, 6]], dtype=float)
    iou = iou_matrix(a, b)
    assert iou.shape == (1, 3)
    assert np.allclose(iou[0], [1.0, 1 / 3, 0.0])


def test_iou_matrix_empty_inputs():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)
    assert iou_matrix(np.zeros((2, 4)), np.zeros((0, 4))).shape == (2, 0)


def test_id_persists_for_a_moving_object():
    tracker = SortTracker(iou_threshold=0.3, max_age=5, min_hits=1)
    ids = [tracker.update([det((10 + 2 * t, 10, 50 + 2 * t, 50))])[0]["id"] for t in range(6)]
    assert len(set(ids)) == 1


def test_two_objects_get_distinct_ids():
    tracker = SortTracker(min_hits=1)
    out = tracker.update([det((0, 0, 40, 40)), det((200, 200, 240, 240))])
    assert len({t["id"] for t in out}) == 2


def test_matching_is_class_aware():
    tracker = SortTracker(min_hits=1)
    first = tracker.update([det((10, 10, 50, 50), label=0)])[0]["id"]
    same_box_other_class = tracker.update([det((10, 10, 50, 50), label=1)])
    assert [t["id"] for t in same_box_other_class if t["label"] == 1] != [first]


def test_track_survives_a_gap_up_to_max_age_then_is_dropped():
    box = (10, 10, 50, 50)
    tracker = SortTracker(max_age=2, min_hits=1)
    first = tracker.update([det(box)])[0]["id"]
    tracker.update([])
    tracker.update([])
    assert tracker.update([det(box)])[0]["id"] == first          # gap of 2 frames, max_age 2

    tracker = SortTracker(max_age=2, min_hits=1)
    first = tracker.update([det(box)])[0]["id"]
    for _ in range(4):
        tracker.update([])
    assert tracker.update([det(box)])[0]["id"] != first          # gap of 4 frames: a new track


def test_min_hits_delays_reporting_of_new_tracks():
    tracker = SortTracker(min_hits=3)
    for _ in range(3):                                           # SORT reports everything in the first frames
        tracker.update([])
    box = (10, 10, 50, 50)
    reported = [len(tracker.update([det(box)])) for _ in range(3)]
    assert reported == [0, 0, 1]
