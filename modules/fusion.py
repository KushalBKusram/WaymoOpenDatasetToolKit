"""Calibrated camera--LiDAR preprocessing used by the early-fusion backend."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from modules.waymo_open_dataset import _CAM_CAL


def project_lidar_xyzi_to_camera(points_xyzi: np.ndarray, camera_calibration,
                                  image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project vehicle-frame XYZI points into one calibrated camera image.

    Returns only points in front of and inside the image.  The source Waymo
    calibration stores camera-to-vehicle, hence the required inverse.
    """
    points = np.asarray(points_xyzi, dtype=np.float32)
    finite = np.isfinite(points).all(axis=1) & (np.abs(points[:, :3]) < 300.0).all(axis=1)
    points = points[finite]
    if not len(points):
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty
    camera_to_vehicle = np.asarray(
        camera_calibration[f'{_CAM_CAL}.extrinsic.transform'], dtype=np.float64
    ).reshape(4, 4)
    vehicle_to_camera = np.linalg.inv(camera_to_vehicle)
    homogeneous = np.concatenate((points[:, :3], np.ones((len(points), 1))), axis=1)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        camera_points = (vehicle_to_camera @ homogeneous.T).T[:, :3]
    finite_camera = np.isfinite(camera_points).all(axis=1)
    camera_points = camera_points[finite_camera]
    points = points[finite_camera]
    # Waymo vehicle coordinates are forward-left-up.  Calibration preserves
    # that convention, so optical depth is the transformed forward (X) axis.
    depth = camera_points[:, 0]
    in_front = depth > 1e-3
    camera_points, depth, intensity = camera_points[in_front], depth[in_front], points[in_front, 3]
    if not len(depth):
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty
    fu = float(camera_calibration[f'{_CAM_CAL}.intrinsic.f_u'])
    fv = float(camera_calibration[f'{_CAM_CAL}.intrinsic.f_v'])
    cu = float(camera_calibration[f'{_CAM_CAL}.intrinsic.c_u'])
    cv = float(camera_calibration[f'{_CAM_CAL}.intrinsic.c_v'])
    u = -camera_points[:, 1] / depth * fu + cu
    v = -camera_points[:, 2] / depth * fv + cv
    height, width = image_shape[:2]
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return (u[valid].astype(np.float32), v[valid].astype(np.float32),
            depth[valid].astype(np.float32), intensity[valid].astype(np.float32))


def lidar_raster(points_xyzi: np.ndarray, camera_calibration, image_shape: tuple[int, int],
                 max_depth: float = 75.0) -> np.ndarray:
    """Create a two-channel nearest-return depth/intensity image in [0, 1]."""
    height, width = image_shape[:2]
    raster = np.zeros((height, width, 2), dtype=np.float32)
    u, v, depth, intensity = project_lidar_xyzi_to_camera(points_xyzi, camera_calibration, image_shape)
    if not len(depth):
        return raster
    # Paint far-to-near so the visible point wins where multiple returns share
    # a pixel.  This is a sparse z-buffer rather than an artificial interpolation.
    order = np.argsort(depth)[::-1]
    x = u[order].astype(np.intp); y = v[order].astype(np.intp)
    raster[y, x, 0] = np.clip(depth[order] / max_depth, 0.0, 1.0)
    raster[y, x, 1] = np.clip(intensity[order], 0.0, 1.0)
    return raster


def build_fusion_tensor(image_bgr: np.ndarray, points_xyzi: np.ndarray, camera_calibration,
                        image_size: int, max_depth: float = 75.0) -> torch.Tensor:
    """Build an RGB + projected-depth + intensity tensor for early fusion."""
    image = cv2.resize(image_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    raster = lidar_raster(points_xyzi, camera_calibration, image_bgr.shape[:2], max_depth=max_depth)
    raster = cv2.resize(raster, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(np.concatenate((image, raster), axis=2)).permute(2, 0, 1).contiguous()
