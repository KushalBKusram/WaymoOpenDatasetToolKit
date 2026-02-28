"""
visualize.py — Notebook-friendly visualisation utilities for Waymo Open Dataset v2.

All public functions return either an annotated BGR numpy array (draw_*)
or a matplotlib Figure (plot_*), so they display cleanly inside Jupyter
via plt.show() / IPython.display.Image without needing separate windows.

Functions
---------
draw_camera_boxes        Overlay 2-D bounding boxes on a camera image.
plot_bev                 Bird's-eye-view scatter of LiDAR points + 3-D box footprints.
build_open3d_scene       Assemble an Open3D PointCloud + box LineSet list for o3d.draw.
project_lidar_to_camera  Project vehicle-frame LiDAR points into pixel space.
draw_lidar_on_camera     Colour-coded LiDAR depth overlay on a camera image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import open3d as o3d
from waymo_open_dataset import v2


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_TYPES = {
    0: 'TYPE_UNKNOWN',
    1: 'TYPE_VEHICLE',
    2: 'TYPE_PEDESTRIAN',
    3: 'TYPE_SIGN',
    4: 'TYPE_CYCLIST',
}

# RGB colours (0–1) for each object class — used in matplotlib and projected
LABEL_COLORS_RGB = {
    'TYPE_UNKNOWN':    (0.50, 0.50, 0.50),
    'TYPE_VEHICLE':    (0.13, 0.86, 0.13),  # green
    'TYPE_PEDESTRIAN': (0.20, 0.60, 1.00),  # sky-blue
    'TYPE_SIGN':       (1.00, 1.00, 0.10),  # yellow
    'TYPE_CYCLIST':    (1.00, 0.50, 0.05),  # orange
}

# BGR uint8 equivalents for OpenCV drawing functions
LABEL_COLORS_BGR = {
    k: tuple(int(c * 255) for c in (r[2], r[1], r[0]))
    for k, r in LABEL_COLORS_RGB.items()
}

CAMERA_NAMES = {
    1: 'FRONT',
    2: 'FRONT_LEFT',
    3: 'FRONT_RIGHT',
    4: 'SIDE_LEFT',
    5: 'SIDE_RIGHT',
}


# ---------------------------------------------------------------------------
# Private geometry helpers
# ---------------------------------------------------------------------------

def _box3d_bev_corners(cx: float, cy: float,
                       sx: float, sy: float,
                       heading: float) -> np.ndarray:
    """Return (4, 2) array of BEV corner coords for one 3-D box."""
    dx, dy   = sx / 2, sy / 2
    local    = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rot      = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return (rot @ local.T).T + np.array([cx, cy])


def _box3d_open3d_lineset(
    cx: float, cy: float, cz: float,
    sx: float, sy: float, sz: float,
    heading: float,
    color: tuple = (0.0, 1.0, 0.0),
) -> o3d.geometry.LineSet:
    """Return an Open3D LineSet wireframe for one oriented 3-D bounding box."""
    dx, dy, dz = sx / 2, sy / 2, sz / 2
    local = np.array([
        [-dx, -dy, -dz], [ dx, -dy, -dz], [ dx,  dy, -dz], [-dx,  dy, -dz],
        [-dx, -dy,  dz], [ dx, -dy,  dz], [ dx,  dy,  dz], [-dx,  dy,  dz],
    ])
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rot3 = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]])
    corners = (rot3 @ local.T).T + np.array([cx, cy, cz])

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],   # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],   # top face
        [0, 4], [1, 5], [2, 6], [3, 7],   # verticals
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines  = o3d.utility.Vector2iVector(edges)
    ls.paint_uniform_color(color)
    return ls


# ---------------------------------------------------------------------------
# Public visualisation functions
# ---------------------------------------------------------------------------

def draw_camera_boxes(
    image: np.ndarray,
    boxes_df,
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """
    Overlay 2-D bounding boxes on a camera frame.

    Args:
        image:     BGR numpy array from toolkit.load_camera_frame().
        boxes_df:  DataFrame from toolkit.load_camera_boxes().  Each row
                   is parsed via v2.CameraBoxComponent.from_dict().
        thickness: Rectangle line thickness in pixels.
        font_scale: Label text size.

    Returns:
        Annotated BGR numpy array (copy of image, not in-place).
    """
    out = image.copy()
    for _, row in boxes_df.iterrows():
        box      = v2.CameraBoxComponent.from_dict(row)
        cx, cy   = box.box.center.x, box.box.center.y
        w, h     = box.box.size.x, box.box.size.y
        x1, y1   = int(cx - w / 2), int(cy - h / 2)
        x2, y2   = int(cx + w / 2), int(cy + h / 2)
        label    = LABEL_TYPES.get(box.type, 'TYPE_UNKNOWN')
        color    = LABEL_COLORS_BGR.get(label, (255, 255, 255))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            out, label.replace('TYPE_', ''),
            (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
            cv2.LINE_AA,
        )
    return out


def plot_bev(
    points_list: list[np.ndarray],
    boxes_df=None,
    range_m: float = 75.0,
    figsize: tuple = (10, 10),
    point_size: float = 0.3,
) -> plt.Figure:
    """
    Bird's-eye-view of LiDAR point cloud with optional 3-D box footprints.

    Args:
        points_list: List of (N, 3) float32 arrays from load_lidar_points().
        boxes_df:    DataFrame from load_lidar_boxes(), or None.
        range_m:     Half-width of the square view in metres (centred on ego).
        figsize:     Matplotlib figure size.
        point_size:  Scatter marker size.

    Returns:
        matplotlib Figure — call plt.show() or display(fig) in a notebook.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    # --- Points coloured by height ---
    if points_list:
        all_pts = np.vstack(points_list)
        z       = all_pts[:, 2]
        z_norm  = np.clip((z - z.min()) / (z.ptp() + 1e-6), 0, 1)
        ax.scatter(
            all_pts[:, 0], all_pts[:, 1],
            c=z_norm, cmap='plasma', s=point_size, linewidths=0, alpha=0.8,
        )

    # --- 3-D box footprints ---
    if boxes_df is not None and len(boxes_df) > 0:
        for _, row in boxes_df.iterrows():
            box     = v2.LiDARBoxComponent.from_dict(row)
            label   = LABEL_TYPES.get(box.type, 'TYPE_UNKNOWN')
            rgb     = LABEL_COLORS_RGB.get(label, (1, 1, 1))
            corners = _box3d_bev_corners(
                box.box.center.x, box.box.center.y,
                box.box.size.x,   box.box.size.y,
                box.box.heading,
            )
            poly = MplPolygon(corners, closed=True,
                              edgecolor=rgb, facecolor=(*rgb, 0.10),
                              linewidth=1.2)
            ax.add_patch(poly)

            # Heading arrow (front of box = first edge midpoint)
            front_mid = (corners[0] + corners[1]) / 2
            ax.annotate(
                '', xy=front_mid,
                xytext=(box.box.center.x, box.box.center.y),
                arrowprops=dict(arrowstyle='->', color=rgb, lw=1.2),
            )

    # --- Legend ---
    legend_handles = [
        mpatches.Patch(color=LABEL_COLORS_RGB[k], label=k.replace('TYPE_', ''))
        for k in LABEL_COLORS_RGB
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              facecolor='#333333', labelcolor='white', fontsize=9)

    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m, range_m)
    ax.set_xlabel('X / m (forward)', color='white')
    ax.set_ylabel('Y / m (left)',    color='white')
    ax.tick_params(colors='white')
    ax.set_aspect('equal')
    ax.set_title('LiDAR — Bird\'s-Eye View', color='white', pad=10)
    fig.tight_layout()
    return fig


def build_open3d_scene(
    points_list: list[np.ndarray],
    boxes_df=None,
) -> list:
    """
    Assemble an Open3D scene: a coloured PointCloud + box LineSet wireframes.

    Usage in a notebook (opens a separate viewer window):
        geometries = build_open3d_scene(points_list, boxes_df)
        o3d.visualization.draw_geometries(geometries)

    Args:
        points_list: List of (N, 3) float32 arrays from load_lidar_points().
        boxes_df:    DataFrame from load_lidar_boxes(), or None.

    Returns:
        List of o3d.geometry objects.
    """
    geometries = []

    # --- Point cloud ---
    if points_list:
        all_pts = np.vstack(points_list).astype(np.float64)
        pcd     = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)

        # Colour by height
        z      = all_pts[:, 2]
        z_norm = np.clip((z - z.min()) / (z.ptp() + 1e-6), 0, 1)
        cmap   = plt.get_cmap('plasma')
        colors = cmap(z_norm)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd)

    # --- 3-D bounding boxes ---
    if boxes_df is not None and len(boxes_df) > 0:
        for _, row in boxes_df.iterrows():
            box   = v2.LiDARBoxComponent.from_dict(row)
            label = LABEL_TYPES.get(box.type, 'TYPE_UNKNOWN')
            color = LABEL_COLORS_RGB.get(label, (1.0, 1.0, 1.0))
            ls    = _box3d_open3d_lineset(
                box.box.center.x, box.box.center.y, box.box.center.z,
                box.box.size.x,   box.box.size.y,   box.box.size.z,
                box.box.heading,  color=color,
            )
            geometries.append(ls)

    # Ego-vehicle marker at origin
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    geometries.append(origin)

    return geometries


def project_lidar_to_camera(
    points_list: list[np.ndarray],
    cam_calib: 'v2.CameraCalibrationComponent',
    image_shape: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project LiDAR points (vehicle frame) into camera pixel coordinates.

    The extrinsic in v2 is camera→vehicle, so we invert it to get
    vehicle→camera, then apply the pinhole intrinsic.

    Args:
        points_list:  List of (N, 3) arrays from load_lidar_points().
        cam_calib:    v2.CameraCalibrationComponent from load_camera_calibration().
        image_shape:  (H, W) to clip to image bounds; None = no clipping.

    Returns:
        u:     Pixel x-coordinates (float64 array).
        v:     Pixel y-coordinates (float64 array).
        depth: Depth in metres along the camera z-axis (float64 array).
    """
    all_pts = np.vstack(points_list).astype(np.float64)  # (N, 3)

    # Vehicle → camera transform
    T_cam_to_veh = np.array(cam_calib.extrinsic.transform).reshape(4, 4)
    T_veh_to_cam = np.linalg.inv(T_cam_to_veh)

    pts_hom = np.hstack([all_pts, np.ones((len(all_pts), 1))])   # (N, 4)
    pts_cam = (T_veh_to_cam @ pts_hom.T).T[:, :3]                # (N, 3)

    # Keep only points in front of the camera
    front = pts_cam[:, 2] > 0
    pts_cam = pts_cam[front]

    fu = cam_calib.intrinsic.f_u
    fv = cam_calib.intrinsic.f_v
    cu = cam_calib.intrinsic.c_u
    cv = cam_calib.intrinsic.c_v

    u     = pts_cam[:, 0] / pts_cam[:, 2] * fu + cu
    v     = pts_cam[:, 1] / pts_cam[:, 2] * fv + cv
    depth = pts_cam[:, 2]

    # Optional: clip to image bounds
    if image_shape is not None:
        H, W = image_shape[:2]
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, depth = u[in_bounds], v[in_bounds], depth[in_bounds]

    return u, v, depth


def draw_lidar_on_camera(
    image: np.ndarray,
    points_list: list[np.ndarray],
    cam_calib: 'v2.CameraCalibrationComponent',
    max_depth: float = 75.0,
    dot_radius: int = 2,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """
    Overlay projected LiDAR points onto a camera image, coloured by depth.

    Args:
        image:      BGR numpy array from load_camera_frame().
        points_list: List of (N, 3) arrays from load_lidar_points().
        cam_calib:  v2.CameraCalibrationComponent from load_camera_calibration().
        max_depth:  Depth value mapped to the far end of the colourmap (metres).
        dot_radius: Radius of each projected point dot (pixels).
        colormap:   OpenCV colourmap constant (default: cv2.COLORMAP_TURBO).

    Returns:
        Annotated BGR numpy array.
    """
    out = image.copy()

    u, v, depth = project_lidar_to_camera(
        points_list, cam_calib, image_shape=image.shape
    )
    if len(u) == 0:
        return out

    # Normalise depth → uint8, apply colourmap
    depth_norm  = np.clip(depth / max_depth, 0.0, 1.0)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    colormap_img = cv2.applyColorMap(
        depth_uint8.reshape(-1, 1), colormap
    ).reshape(-1, 3)   # (N, 3) BGR

    for (px, py, col) in zip(u.astype(int), v.astype(int), colormap_img):
        cv2.circle(out, (px, py), dot_radius, col.tolist(), -1)

    return out
