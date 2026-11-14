"""Per-detection 3-D positions from TOP-LiDAR points, for LiDAR-gated tracking.

For one camera frame, the LiDAR points are projected into the camera image once (`project_frame`). For each 2-D
detection `estimate_box` takes the points that fall in the central part of the box, drops likely ground points,
keeps the points around the dominant depth and returns their robust centroid in the vehicle frame (median x, y, z)
with the number of points kept as a confidence. `to_world` turns a vehicle-frame position into the world frame with
the frame's vehicle pose, so positions of the same static object agree across frames despite ego motion.

Numpy only (the camera projection repeats the maths of `modules.fusion` without importing torch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAM_CAL = "[CameraCalibrationComponent]"

INNER = 0.6                 # central fraction of the box (width and height) whose points are used
GROUND_Z = -1.5             # vehicle-frame height below which points are treated as ground (sensor is about 2 m up)
DEPTH_BAND = 2.5            # metres around the dominant depth that are kept
MAX_RANGE = 120.0


@dataclass
class CameraModel:
    """Pinhole camera with a vehicle-to-camera transform (Waymo vehicle frame: x forward, y left, z up)."""
    vehicle_to_camera: np.ndarray
    fu: float
    fv: float
    cu: float
    cv: float
    width: int
    height: int


def camera_model_from_row(row) -> CameraModel:
    """Build a CameraModel from a Waymo v2 camera_calibration row (a pandas Series)."""
    camera_to_vehicle = np.asarray(row[f"{CAM_CAL}.extrinsic.transform"], dtype=np.float64).reshape(4, 4)
    return CameraModel(np.linalg.inv(camera_to_vehicle), float(row[f"{CAM_CAL}.intrinsic.f_u"]),
                       float(row[f"{CAM_CAL}.intrinsic.f_v"]), float(row[f"{CAM_CAL}.intrinsic.c_u"]),
                       float(row[f"{CAM_CAL}.intrinsic.c_v"]), int(row[f"{CAM_CAL}.width"]),
                       int(row[f"{CAM_CAL}.height"]))


@dataclass
class FrameProjection:
    """LiDAR points of one frame that project into the image: pixel coordinates, depth and vehicle-frame xyz."""
    u: np.ndarray
    v: np.ndarray
    depth: np.ndarray
    xyz: np.ndarray


def project_frame(points_xyz: np.ndarray, camera: CameraModel) -> FrameProjection:
    """Project vehicle-frame points (N, >=3) into the camera; keeps points in front of it and inside the image."""
    points = np.asarray(points_xyz, dtype=np.float64)[:, :3]
    ok = np.isfinite(points).all(axis=1) & (np.abs(points) < 300.0).all(axis=1)
    points = points[ok]
    if not len(points):
        empty = np.empty(0)
        return FrameProjection(empty, empty, empty, np.empty((0, 3)))
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    camera_points = (camera.vehicle_to_camera @ homogeneous.T).T[:, :3]
    depth = camera_points[:, 0]                      # optical depth is the transformed forward axis
    front = depth > 1e-3
    camera_points, depth, points = camera_points[front], depth[front], points[front]
    u = -camera_points[:, 1] / depth * camera.fu + camera.cu
    v = -camera_points[:, 2] / depth * camera.fv + camera.cv
    inside = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    return FrameProjection(u[inside], v[inside], depth[inside], points[inside])


def estimate_box(frame: FrameProjection, box: list[float] | np.ndarray, inner: float = INNER,
                 band: float = DEPTH_BAND) -> tuple[np.ndarray | None, int]:
    """Robust vehicle-frame centroid of the LiDAR points behind the central part of a 2-D box.

    Returns (xyz or None, number of points kept). None when no point survives.
    """
    if not len(frame.depth):
        return None, 0
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy, half_w, half_h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * inner / 2, (y2 - y1) * inner / 2
    chosen = ((frame.u >= cx - half_w) & (frame.u <= cx + half_w) & (frame.v >= cy - half_h) & (frame.v <= cy + half_h)
              & (frame.xyz[:, 2] > GROUND_Z) & (frame.depth < MAX_RANGE))
    if not chosen.any():
        return None, 0
    depth, xyz = frame.depth[chosen], frame.xyz[chosen]
    # The object's points are the foreground cluster; points through gaps belong to what is behind it.
    # Take the densest depth window of width `band` starting from the nearer half of the points.
    order = np.argsort(depth)
    depth, xyz = depth[order], xyz[order]
    best_start, best_count = 0, 0
    for start in range(len(depth)):
        count = int(np.searchsorted(depth, depth[start] + band, side="right") - start)
        if count > best_count:
            best_start, best_count = start, count
    keep = slice(best_start, best_start + best_count)
    return np.median(xyz[keep], axis=0), best_count


def to_world(xyz_vehicle: np.ndarray, world_from_vehicle: np.ndarray) -> np.ndarray:
    """Vehicle-frame position(s) -> world frame with the (4, 4) world_from_vehicle pose."""
    xyz = np.atleast_2d(np.asarray(xyz_vehicle, dtype=np.float64))
    return (xyz @ world_from_vehicle[:3, :3].T + world_from_vehicle[:3, 3]).reshape(np.shape(xyz_vehicle))


def from_world(xyz_world: np.ndarray, world_from_vehicle: np.ndarray) -> np.ndarray:
    """Inverse of `to_world`."""
    xyz = np.atleast_2d(np.asarray(xyz_world, dtype=np.float64))
    return ((xyz - world_from_vehicle[:3, 3]) @ world_from_vehicle[:3, :3]).reshape(np.shape(xyz_world))


def attach_lidar3d(toolkit, frames: dict, camera: int, run_dir, min_cache: bool = True) -> float:
    """Add 'xyz_world' (3-vector or None), 'xyz_vehicle' and 'n3d' to every detection in `frames`
    (as returned by `track_eval.collect`).

    Estimates are cached in `<run_dir>/tracking/lidar3d.npz`, per segment: arrays for all detections of the
    segment in order, plus the per-frame detection counts; a segment is recomputed when the counts change.
    Returns the seconds spent estimating (0 when everything was cached).
    """
    import time
    from pathlib import Path

    cache_path = Path(run_dir) / "tracking" / "lidar3d.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    spent, dirty = 0.0, False
    for segment, segment_frames in frames.items():
        counts = np.array([len(dets) for _, dets in segment_frames], dtype=np.int64)
        if f"{segment}|world" not in cache or not np.array_equal(cache.get(f"{segment}|counts"), counts):
            started = time.time()
            toolkit.assign_segment(segment)
            model = camera_model_from_row(toolkit.load_camera_calibration(camera))
            poses = dict(toolkit.load_all_vehicle_poses())
            timestamps = sorted(int(t) for t in toolkit.get_timestamps())
            world, vehicle, kept = [], [], []
            for timestamp, (_, dets) in zip(timestamps, segment_frames):
                if not dets:
                    continue
                sets = toolkit.load_lidar_points_xyzi(timestamp, top_only=True)
                points = max(sets, key=len) if sets else np.zeros((0, 4), np.float32)
                projection = project_frame(points, model)
                for det in dets:
                    xyz, n = estimate_box(projection, det["box"])
                    if xyz is None:
                        world.append(np.full(3, np.nan)); vehicle.append(np.full(3, np.nan)); kept.append(0)
                    else:
                        world.append(to_world(xyz, poses[timestamp])); vehicle.append(xyz); kept.append(n)
            cache[f"{segment}|world"] = np.array(world, dtype=np.float32).reshape(-1, 3)
            cache[f"{segment}|vehicle"] = np.array(vehicle, dtype=np.float32).reshape(-1, 3)
            cache[f"{segment}|n"] = np.array(kept, dtype=np.int32)
            cache[f"{segment}|counts"] = counts
            spent += time.time() - started
            dirty = True
        offset = 0
        for _, dets in segment_frames:
            for det in dets:
                w, n = cache[f"{segment}|world"][offset], int(cache[f"{segment}|n"][offset])
                det["xyz_world"] = None if np.isnan(w[0]) else w.astype(np.float64)
                det["xyz_vehicle"] = None if np.isnan(w[0]) else cache[f"{segment}|vehicle"][offset].astype(np.float64)
                det["n3d"] = n
                offset += 1
    if dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **cache)
    return spent
