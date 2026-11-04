from track_replay import match, replay_segment

CLASSES = ["vehicle", "pedestrian"]
BOX = (10, 10, 60, 60)


def gt(box=BOX, obj_id="a", label=0):
    return {"id": obj_id, "label": label, "box": list(box)}


def det(box=BOX, label=0, score=0.9):
    return {"box": list(box), "label": label, "score": score}


def settings(**overrides):
    return {"score_thresh": 0.25, "iou_threshold": 0.3, "max_age": 5, "min_hits": 1, **overrides}


def events_of(frames, kind, **kw):
    _, events = replay_segment(frames, CLASSES, settings(**kw), 0.5)
    return [e for e in events if e["type"] == kind]


def test_match_pairs_overlapping_boxes_only():
    track = {"id": 7, "box": list(BOX), "label": 0}
    assert match([gt()], [track], {}, 0.5) == {0: 0}
    assert match([gt()], [{"id": 8, "box": [300, 300, 340, 340], "label": 0}], {}, 0.5) == {}
    assert match([], [track], {}, 0.5) == {}


def test_miss_with_no_detection_is_a_detector_miss():
    misses = events_of([([gt()], [])], "miss")
    assert [m["cause"] for m in misses] == ["detector"]


def test_miss_with_low_score_detection_is_a_gate_miss():
    misses = events_of([([gt()], [det(score=0.1)])], "miss")
    assert [m["cause"] for m in misses] == ["gate"]
    # the low-score detection is inside the cache but below the gate, so the tracker never sees it


def test_unconfirmed_track_is_an_association_miss():
    frames = [([], [])] * 3 + [([gt()], [det()])]                # frames 1-3 empty, object appears on frame 4
    misses = events_of(frames, "miss", min_hits=3)
    assert [m["cause"] for m in misses] == ["association"]


def test_id_switch_after_a_gap_is_counted():
    seen = ([gt()], [det()])
    lost = ([gt()], [])
    frames = [seen] * 3 + [lost] * 3 + [seen]                    # max_age 1 drops the track during the gap
    switches = events_of(frames, "switch", max_age=1)
    assert len(switches) == 1
    assert switches[0]["gap"] > 1
    assert switches[0]["from"] != switches[0]["to"]


def test_continuous_tracking_has_no_switches_or_misses():
    frames = [([gt()], [det()])] * 5
    assert events_of(frames, "switch") == []
    assert events_of(frames, "miss") == []
