"""LiDAR 3-D gating: depth estimate from points, world-frame conversion and the tracker gate."""

import numpy as np

from modules.lidar_gate import CameraModel, estimate_box, from_world, project_frame, to_world
from modules.tracking import SortTracker

CAMERA = CameraModel(vehicle_to_camera=np.eye(4), fu=1000.0, fv=1000.0, cu=960.0, cv=640.0, width=1920, height=1280)


def points_at(x: float, y: float, z: float, count: int = 30, spread: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.column_stack([x + rng.uniform(-spread, spread, count), y + rng.uniform(-spread, spread, count),
                            z + rng.uniform(-spread, spread, count), np.ones(count)])


def test_world_conversion_round_trips():
    pose = np.eye(4)
    pose[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])    # 90 degree yaw
    pose[:3, 3] = [100.0, -20.0, 3.0]
    xyz = np.array([12.0, 3.0, 0.5])
    world = to_world(xyz, pose)
    assert np.allclose(world, [97.0, -8.0, 3.5])
    assert np.allclose(from_world(world, pose), xyz)
    many = np.random.default_rng(1).normal(size=(5, 3))
    assert np.allclose(from_world(to_world(many, pose), pose), many)


def test_estimate_uses_the_foreground_cluster_and_ignores_ground_and_background():
    # Object 20 m ahead (pixel (960, 615)); background wall 45 m behind the same pixels; ground in front of the box.
    object_points = points_at(20.0, 0.0, 0.5)
    background = points_at(45.0, 0.0, 0.5, count=20)
    ground = points_at(12.0, 0.0, -1.9, count=20)
    frame = project_frame(np.vstack([object_points, background, ground]), CAMERA)
    xyz, n = estimate_box(frame, [900.0, 540.0, 1020.0, 700.0])
    assert xyz is not None and n >= 25
    assert abs(xyz[0] - 20.0) < 0.5 and abs(xyz[1]) < 0.5 and abs(xyz[2] - 0.5) < 0.5


def test_estimate_returns_none_without_points():
    frame = project_frame(points_at(20.0, 0.0, 0.5), CAMERA)
    xyz, n = estimate_box(frame, [100.0, 100.0, 160.0, 160.0])           # nothing projects there
    assert xyz is None and n == 0
    xyz, n = estimate_box(project_frame(np.zeros((0, 4)), CAMERA), [0.0, 0.0, 10.0, 10.0])
    assert xyz is None and n == 0


def detection(box, xyz=None, n3d=30):
    det = {"box": box, "label": 0, "score": 0.9}
    if xyz is not None:
        det.update({"xyz_world": np.array(xyz, dtype=float), "n3d": n3d})
    return det


def two_overlapping_then_one(gate: float, **kwargs) -> int:
    """Two boxes overlap in the image but are 10 m apart in depth; one detection then continues the far object."""
    tracker = SortTracker(iou_threshold=0.3, max_age=5, min_hits=1, lidar_gate=gate, **kwargs)
    tracker.update([detection([100, 100, 200, 200], (20, 0, 0)), detection([110, 100, 210, 200], (30, 0, 0))])
    out = tracker.update([detection([100, 100, 200, 200], (30, 0, 0))])    # exactly where track 1 was, but 30 m away
    return out[0]["id"]


def test_gate_keeps_ids_when_boxes_overlap_but_depth_differs():
    assert two_overlapping_then_one(gate=0.0) == 1          # plain IoU takes the identical box (the near object)
    assert two_overlapping_then_one(gate=3.0) == 2          # the gate sends it to the object 30 m away


def test_gate_is_inactive_without_estimates_or_with_too_few_points():
    tracker = SortTracker(iou_threshold=0.3, max_age=5, min_hits=1, lidar_gate=3.0)
    tracker.update([detection([100, 100, 200, 200]), detection([110, 100, 210, 200])])    # no 3-D estimates
    assert tracker.update([detection([100, 100, 200, 200], (30, 0, 0))])[0]["id"] == 1
    assert two_overlapping_then_one(gate=3.0, lidar_min_points=100) == 1                  # 30 points < 100: ignored


def test_cost_blend_prefers_the_depth_consistent_track():
    assert two_overlapping_then_one(gate=0.0, lidar_weight=0.0) == 1
    assert two_overlapping_then_one(gate=0.0, lidar_weight=0.9) == 2


def test_no_lidar_information_leaves_the_tracker_unchanged():
    rng = np.random.default_rng(3)
    plain, with_keys = SortTracker(0.2, 10, 1), SortTracker(0.2, 10, 1)
    for _ in range(25):
        dets = [{"box": list(map(float, (x, y, x + w, y + h))), "label": int(rng.integers(0, 2)), "score": 0.9}
                for x, y, w, h in rng.uniform([0, 0, 30, 30], [400, 300, 80, 80], size=(6, 4))]
        with_xyz = [{**d, "xyz_world": np.array([rng.normal(), rng.normal(), 0.0]), "n3d": 40} for d in dets]
        assert plain.update(dets) == with_keys.update(with_xyz)        # gate and weight both 0: 3-D data is ignored
