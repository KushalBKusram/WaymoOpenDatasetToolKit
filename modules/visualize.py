"""
visualize.py — Notebook-friendly visualisation utilities for Waymo v2.

No waymo-open-dataset pip package required. All DataFrame columns are accessed
directly by their documented v2 Parquet names:
  key fields      ->  key.<field>
  component data  ->  [ComponentClassName].<field>.<subfield>

All public functions return either an annotated BGR numpy array (draw_*)
or a matplotlib Figure (plot_*), so they display correctly in Jupyter
via plt.show() / IPython.display without opening separate windows.

Functions
---------
draw_camera_boxes        Overlay 2-D bounding boxes on a camera image.
plot_bev                 Bird's-eye-view scatter of LiDAR points + 3-D boxes.
build_open3d_scene       Assemble an Open3D PointCloud + box LineSet list.
project_lidar_to_camera  Project vehicle-frame LiDAR points into pixel space.
draw_lidar_on_camera     Colour-coded LiDAR depth overlay on a camera image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
import open3d as o3d


# ---------------------------------------------------------------------------
# Column-name prefixes (must match waymo_open_dataset.py)
# ---------------------------------------------------------------------------
_C_BOX = '[CameraBoxComponent]'
_L_BOX = '[LiDARBoxComponent]'
_CAM_CAL = '[CameraCalibrationComponent]'


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

# RGB colours (0-1) per object class
LABEL_COLORS_RGB = {
    'TYPE_UNKNOWN':    (0.50, 0.50, 0.50),
    'TYPE_VEHICLE':    (0.13, 0.86, 0.13),
    'TYPE_PEDESTRIAN': (0.20, 0.60, 1.00),
    'TYPE_SIGN':       (1.00, 1.00, 0.10),
    'TYPE_CYCLIST':    (1.00, 0.50, 0.05),
}

# BGR uint8 equivalents for OpenCV
LABEL_COLORS_BGR = {
    k: tuple(int(c * 255) for c in (rgb[2], rgb[1], rgb[0]))
    for k, rgb in LABEL_COLORS_RGB.items()
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

def _box3d_bev_corners(cx, cy, sx, sy, heading) -> np.ndarray:
    """Return (4, 2) BEV corner coords for one oriented 3-D box."""
    dx, dy = sx / 2, sy / 2
    local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return (rot @ local.T).T + np.array([cx, cy])


def _box3d_open3d_lineset(cx, cy, cz, sx, sy, sz, heading,
                          color=(0.0, 1.0, 0.0)) -> o3d.geometry.LineSet:
    """Return an Open3D LineSet wireframe for one oriented 3-D bounding box."""
    dx, dy, dz = sx / 2, sy / 2, sz / 2
    local = np.array([
        [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
        [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz],
    ])
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rot3 = np.array([
        [cos_h, -sin_h, 0],
        [sin_h, cos_h, 0],
        [0, 0, 1],
    ])
    corners = (rot3 @ local.T).T + np.array([cx, cy, cz])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.paint_uniform_color(color)
    return ls


# ---------------------------------------------------------------------------
# Public visualisation functions
# ---------------------------------------------------------------------------

def draw_camera_boxes(image: np.ndarray, boxes_df,
                      thickness: int = 2,
                      font_scale: float = 0.55) -> np.ndarray:
    """
    Overlay 2-D bounding boxes on a camera frame.

    Args:
        image:     BGR numpy array from toolkit.load_camera_frame().
        boxes_df:  DataFrame from toolkit.load_camera_boxes().
        thickness: Rectangle line thickness in pixels.
        font_scale: Label text size.

    Returns:
        Annotated BGR numpy array (copy -- not in-place).
    """
    out = image.copy()
    for _, row in boxes_df.iterrows():
        cx = float(row[f'{_C_BOX}.box.center.x'])
        cy = float(row[f'{_C_BOX}.box.center.y'])
        w = float(row[f'{_C_BOX}.box.size.x'])
        h = float(row[f'{_C_BOX}.box.size.y'])
        label = LABEL_TYPES.get(int(row[f'{_C_BOX}.type']), 'TYPE_UNKNOWN')
        color = LABEL_COLORS_BGR.get(label, (255, 255, 255))

        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            out, label.replace('TYPE_', ''),
            (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
            cv2.LINE_AA,
        )
    return out


def plot_bev(points_list: list, boxes_df=None,
             range_m: float = 75.0,
             figsize: tuple = (10, 10),
             point_size: float = 0.3) -> plt.Figure:
    """
    Bird's-eye-view of the LiDAR point cloud with optional 3-D box footprints.

    Args:
        points_list: List of (N, 3) float32 arrays from load_lidar_points().
        boxes_df:    DataFrame from load_lidar_boxes(), or None.
        range_m:     Half-width of the square view in metres (centred on ego).
        figsize:     Matplotlib figure size.
        point_size:  Scatter marker size.

    Returns:
        matplotlib Figure -- call plt.show() or display(fig) in a notebook.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    # --- Points coloured by height ---
    if points_list:
        all_pts = np.vstack(points_list)
        z = all_pts[:, 2]
        z_norm = np.clip((z - z.min()) / (z.ptp() + 1e-6), 0, 1)
        ax.scatter(
            all_pts[:, 0], all_pts[:, 1],
            c=z_norm, cmap='plasma', s=point_size, linewidths=0, alpha=0.8,
        )

    # --- 3-D box footprints ---
    if boxes_df is not None and len(boxes_df) > 0:
        for _, row in boxes_df.iterrows():
            label = LABEL_TYPES.get(int(row[f'{_L_BOX}.type']), 'TYPE_UNKNOWN')
            rgb = LABEL_COLORS_RGB.get(label, (1, 1, 1))
            corners = _box3d_bev_corners(
                float(row[f'{_L_BOX}.box.center.x']),
                float(row[f'{_L_BOX}.box.center.y']),
                float(row[f'{_L_BOX}.box.size.x']),
                float(row[f'{_L_BOX}.box.size.y']),
                float(row[f'{_L_BOX}.box.heading']),
            )
            poly = MplPolygon(corners, closed=True,
                              edgecolor=rgb, facecolor=(*rgb, 0.10),
                              linewidth=1.2)
            ax.add_patch(poly)

            # Heading arrow pointing toward the front edge mid-point
            front_mid = (corners[0] + corners[1]) / 2
            ax.annotate(
                '', xy=front_mid,
                xytext=(float(row[f'{_L_BOX}.box.center.x']),
                        float(row[f'{_L_BOX}.box.center.y'])),
                arrowprops=dict(arrowstyle='->', color=rgb, lw=1.2),
            )

    # Legend
    handles = [
        mpatches.Patch(color=LABEL_COLORS_RGB[k], label=k.replace('TYPE_', ''))
        for k in LABEL_COLORS_RGB
    ]
    ax.legend(handles=handles, loc='upper right',
              facecolor='#333333', labelcolor='white', fontsize=9)

    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m, range_m)
    ax.set_xlabel('X / m (forward)', color='white')
    ax.set_ylabel('Y / m (left)', color='white')
    ax.tick_params(colors='white')
    ax.set_aspect('equal')
    ax.set_title("LiDAR -- Bird's-Eye View", color='white', pad=10)
    fig.tight_layout()
    return fig


def build_open3d_scene(points_list: list, boxes_df=None) -> list:
    """
    Assemble an Open3D scene: coloured PointCloud + box LineSet wireframes.

    Usage (opens a separate interactive window):
        geometries = build_open3d_scene(points_list, boxes_df)
        o3d.visualization.draw_geometries(geometries)

    Args:
        points_list: List of (N, 3) float32 arrays from load_lidar_points().
        boxes_df:    DataFrame from load_lidar_boxes(), or None.

    Returns:
        List of o3d.geometry objects.
    """
    geometries = []

    # Point cloud coloured by height
    if points_list:
        all_pts = np.vstack(points_list).astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        z = all_pts[:, 2]
        z_norm = np.clip((z - z.min()) / (z.ptp() + 1e-6), 0, 1)
        colors = plt.get_cmap('plasma')(z_norm)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd)

    # 3-D bounding box wireframes
    if boxes_df is not None and len(boxes_df) > 0:
        for _, row in boxes_df.iterrows():
            label = LABEL_TYPES.get(int(row[f'{_L_BOX}.type']), 'TYPE_UNKNOWN')
            color = LABEL_COLORS_RGB.get(label, (1.0, 1.0, 1.0))
            ls = _box3d_open3d_lineset(
                float(row[f'{_L_BOX}.box.center.x']),
                float(row[f'{_L_BOX}.box.center.y']),
                float(row[f'{_L_BOX}.box.center.z']),
                float(row[f'{_L_BOX}.box.size.x']),
                float(row[f'{_L_BOX}.box.size.y']),
                float(row[f'{_L_BOX}.box.size.z']),
                float(row[f'{_L_BOX}.box.heading']),
                color=color,
            )
            geometries.append(ls)

    # Ego-vehicle marker at origin
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.0
    ))
    return geometries


def project_lidar_to_camera(
    points_list: list,
    cam_calib_row,
    image_shape: tuple | None = None,
) -> tuple:
    """
    Project LiDAR points (vehicle frame) into camera pixel coordinates.

    The camera extrinsic in v2 is camera->vehicle, so we invert it to get
    vehicle->camera, then apply the pinhole intrinsic model.

    Args:
        points_list:    List of (N, 3) arrays from load_lidar_points().
        cam_calib_row:  pandas Series from load_camera_calibration().
        image_shape:    (H, W) to filter points outside the image; None=no clip.

    Returns:
        u, v: pixel coordinates (float64 arrays).
        depth: depth in metres along camera z-axis (float64 array).
    """
    all_pts = np.vstack(points_list).astype(np.float64)

    cam_to_veh = np.array(
        cam_calib_row[f'{_CAM_CAL}.extrinsic.transform'],
        dtype=np.float64
    ).reshape(4, 4)
    veh_to_cam = np.linalg.inv(cam_to_veh)

    pts_hom = np.hstack([all_pts, np.ones((len(all_pts), 1))])
    pts_cam = (veh_to_cam @ pts_hom.T).T[:, :3]

    front = pts_cam[:, 2] > 0
    pts_cam = pts_cam[front]

    fu = float(cam_calib_row[f'{_CAM_CAL}.intrinsic.f_u'])
    fv = float(cam_calib_row[f'{_CAM_CAL}.intrinsic.f_v'])
    cu = float(cam_calib_row[f'{_CAM_CAL}.intrinsic.c_u'])
    cv = float(cam_calib_row[f'{_CAM_CAL}.intrinsic.c_v'])

    u = pts_cam[:, 0] / pts_cam[:, 2] * fu + cu
    v = pts_cam[:, 1] / pts_cam[:, 2] * fv + cv
    depth = pts_cam[:, 2]

    if image_shape is not None:
        height, width = image_shape[:2]
        in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v, depth = u[in_bounds], v[in_bounds], depth[in_bounds]

    return u, v, depth


def draw_lidar_on_camera(
    image: np.ndarray,
    points_list: list,
    cam_calib_row,
    max_depth: float = 75.0,
    dot_radius: int = 2,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """
    Overlay depth-coloured LiDAR points onto a camera image.

    Args:
        image:          BGR numpy array from load_camera_frame().
        points_list:    List of (N, 3) arrays from load_lidar_points().
        cam_calib_row:  pandas Series from load_camera_calibration().
        max_depth:      Depth clamped to this value before colour-mapping (m).
        dot_radius:     Radius of each projected dot (pixels).
        colormap:       OpenCV colourmap constant (default TURBO).

    Returns:
        Annotated BGR numpy array.
    """
    out = image.copy()
    u, v, depth = project_lidar_to_camera(
        points_list, cam_calib_row, image_shape=image.shape
    )
    if len(u) == 0:
        return out

    depth_norm = np.clip(depth / max_depth, 0.0, 1.0)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    colors = cv2.applyColorMap(
        depth_uint8.reshape(-1, 1), colormap
    ).reshape(-1, 3)

    for px, py, col in zip(u.astype(int), v.astype(int), colors):
        cv2.circle(out, (px, py), dot_radius, col.tolist(), -1)

    return out


# ---------------------------------------------------------------------------
# Column-name prefix exposed for notebooks that need direct access
# ---------------------------------------------------------------------------
_L_BOX_PREFIX = _L_BOX
_C_BOX_PREFIX = _C_BOX
_CAM_CAL_PREFIX = _CAM_CAL
