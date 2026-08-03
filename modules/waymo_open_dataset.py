"""
waymo_open_dataset.py — Waymo Open Dataset v2 (Parquet format) toolkit.

All Parquet column names follow the official v2 naming convention:
  key fields      →  key.<field>
  component data  →  [ComponentClassName].<field>.<subfield>

No waymo-open-dataset pip package is required. Data is read with
dask/pandas/pyarrow and streamed directly from GCS.
LiDAR range images are decoded with NumPy; no Waymo devkit is required.

GCS layout:
  gs://waymo_open_dataset_v_2_0_0/<split>/<component>/<context_name>.parquet
"""

from __future__ import annotations

import os
import pickle

import cv2
import numpy as np
import pandas as pd
import dask.dataframe as dd
import gcsfs



# ---------------------------------------------------------------------------
# Parquet column-name constants
# Column names follow: [ComponentClassName].field.subfield
# Confirmed from waymo-open-dataset/src/waymo_open_dataset/v2/component.py
# ---------------------------------------------------------------------------

# camera_image component
_C_IMG = '[CameraImageComponent]'

# camera_box component
_C_BOX = '[CameraBoxComponent]'

# lidar_box component
_L_BOX = '[LiDARBoxComponent]'

# lidar component (range images)
_L = '[LiDARComponent]'

# lidar_calibration component
_L_CAL = '[LiDARCalibrationComponent]'

# camera_calibration component
_CAM_CAL = '[CameraCalibrationComponent]'

# camera_segmentation component (2-D panoptic segmentation)
# Note: only available for segments annotated for the Panoptic Video Seg challenge
_CAM_SEG = '[CameraSegmentationLabelComponent]'

# lidar_segmentation component (3-D semantic segmentation, per-point labels)
# Note: only available for segments annotated for the 3-D Semantic Seg challenge
_L_SEG = '[LiDARSegmentationLabelComponent]'

# camera_hkp component (2-D human keypoints, 17-point COCO-style skeleton)
# Note: only available for segments annotated for the Pose Estimation challenge
_CAM_HKP = '[CameraHumanKeypointsComponent]'

# projected_lidar_box component (3-D LiDAR boxes reprojected into camera images)
_PROJ_BOX = '[ProjectedLiDARBoxComponent]'

# vehicle_pose component (ego-vehicle world pose per frame — always present)
_VEH_POSE = '[VehiclePoseComponent]'


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

CAMERA_NAMES = {
    1: 'FRONT',
    2: 'FRONT_LEFT',
    3: 'FRONT_RIGHT',
    4: 'SIDE_LEFT',
    5: 'SIDE_RIGHT',
}

LABEL_TYPES = {
    0: 'TYPE_UNKNOWN',
    1: 'TYPE_VEHICLE',
    2: 'TYPE_PEDESTRIAN',
    3: 'TYPE_SIGN',
    4: 'TYPE_CYCLIST',
}

# YOLO class mapping — TYPE_UNKNOWN excluded (no label written for those boxes)
YOLO_CLASS_MAP = {
    1: 0,   # TYPE_VEHICLE    → 0
    2: 1,   # TYPE_PEDESTRIAN → 1
    4: 2,   # TYPE_CYCLIST    → 2
    3: 3,   # TYPE_SIGN       → 3
}
YOLO_CLASS_NAMES = ['vehicle', 'pedestrian', 'cyclist', 'sign']

# 3-D LiDAR detection class mapping (vehicle / pedestrian / cyclist only;
# sign is excluded because it rarely appears in the top-view BEV)
LIDAR_DET_CLASS_MAP: dict[int, int] = {
    1: 0,   # TYPE_VEHICLE    → 0
    2: 1,   # TYPE_PEDESTRIAN → 1
    4: 2,   # TYPE_CYCLIST    → 2
}
LIDAR_DET_CLASS_NAMES: list[str] = ['vehicle', 'pedestrian', 'cyclist']

# Semantic class labels — lidar_segmentation (23 classes)
# Source: Waymo Open Dataset v2 LiDAR semantic segmentation label spec
LIDAR_SEG_LABELS: dict[int, str] = {
    0:  'TYPE_UNDEFINED',
    1:  'TYPE_CAR',
    2:  'TYPE_TRUCK',
    3:  'TYPE_BUS',
    4:  'TYPE_OTHER_VEHICLE',
    5:  'TYPE_MOTORCYCLIST',
    6:  'TYPE_BICYCLIST',
    7:  'TYPE_PEDESTRIAN',
    8:  'TYPE_SIGN',
    9:  'TYPE_TRAFFIC_LIGHT',
    10: 'TYPE_POLE',
    11: 'TYPE_CONSTRUCTION_CONE',
    12: 'TYPE_BICYCLE',
    13: 'TYPE_MOTORCYCLE',
    14: 'TYPE_BUILDING',
    15: 'TYPE_VEGETATION',
    16: 'TYPE_TREE_TRUNK',
    17: 'TYPE_CURB',
    18: 'TYPE_ROAD',
    19: 'TYPE_LANE_MARKER',
    20: 'TYPE_OTHER_GROUND',
    21: 'TYPE_WALKABLE',
    22: 'TYPE_SIDEWALK',
}

# Semantic class labels — camera_segmentation (29 classes, panoptic)
# Source: Waymo Open Dataset v2 camera segmentation label spec
CAM_SEG_LABELS: dict[int, str] = {
    0:  'TYPE_UNDEFINED',
    1:  'TYPE_EGO_VEHICLE',
    2:  'TYPE_CAR',
    3:  'TYPE_TRUCK',
    4:  'TYPE_BUS',
    5:  'TYPE_OTHER_LARGE_VEHICLE',
    6:  'TYPE_BICYCLE',
    7:  'TYPE_MOTORCYCLE',
    8:  'TYPE_TRAILER',
    9:  'TYPE_PEDESTRIAN',
    10: 'TYPE_CYCLIST',
    11: 'TYPE_MOTORCYCLIST',
    12: 'TYPE_BIRD',
    13: 'TYPE_GROUND_ANIMAL',
    14: 'TYPE_CONSTRUCTION_CONE_POLE',
    15: 'TYPE_POLE',
    16: 'TYPE_PEDESTRIAN_OBJECT',
    17: 'TYPE_SIGN',
    18: 'TYPE_TRAFFIC_LIGHT',
    19: 'TYPE_BUILDING',
    20: 'TYPE_ROAD',
    21: 'TYPE_LANE_MARKER',
    22: 'TYPE_ROAD_MARKER',
    23: 'TYPE_SIDEWALK',
    24: 'TYPE_VEGETATION',
    25: 'TYPE_SKY',
    26: 'TYPE_GROUND',
    27: 'TYPE_DYNAMIC',
    28: 'TYPE_STATIC',
}


# ---------------------------------------------------------------------------
# ToolKit
# ---------------------------------------------------------------------------

class ToolKit:
    """High-level interface for the Waymo Open Dataset v2 (Parquet format).

    Supports two usage modes:
      * Extraction mode  — writes images, labels and point clouds to disk
                           (used by main.py and batch scripts)
      * Notebook mode    — load_* methods return in-memory numpy arrays /
                           DataFrames suitable for EDA and visualisation

    No waymo-open-dataset pip package required — columns are accessed directly
    by their documented v2 Parquet names.
    """

    GCS_BUCKET = 'waymo_open_dataset_v_2_0_0'

    def __init__(self, split: str = 'training', save_dir: str = './output',
                 cache_dir: str | None = './.waymo_cache'):
        assert split in ('training', 'validation', 'testing'), (
            f"split must be 'training', 'validation', or 'testing',"
            f" got '{split}'"
        )
        self.split = split
        self.save_dir = save_dir
        self.cache_dir = cache_dir
        self.context_name: str | None = None
        self._df_cache: dict[str, pd.DataFrame] = {}
        self._setup_dirs()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _setup_dirs(self):
        self.camera_images_dir = os.path.join(
            self.save_dir, 'camera', 'images')
        self.camera_labels_dir = os.path.join(
            self.save_dir, 'camera', 'labels')
        self.lidar_points_dir = os.path.join(
            self.save_dir, 'lidar', 'points')
        self.lidar_labels_dir = os.path.join(
            self.save_dir, 'lidar', 'labels')
        for d in (self.camera_images_dir, self.camera_labels_dir,
                  self.lidar_points_dir, self.lidar_labels_dir):
            os.makedirs(d, exist_ok=True)

    def _gcs_path(self, component: str) -> str:
        """Return a local cached component when available, otherwise GCS."""
        if self.cache_dir:
            local_path = os.path.join(
                self.cache_dir, self.split, component,
                f'{self.context_name}.parquet',
            )
            if os.path.isfile(local_path):
                return local_path
        return (f'gs://{self.GCS_BUCKET}/{self.split}'
                f'/{component}/{self.context_name}.parquet')

    def _read(self, component: str, columns=None, filters=None) -> dd.DataFrame:
        """Return a lazy Dask DataFrame for one component of the segment."""
        return dd.read_parquet(self._gcs_path(component), columns=columns, filters=filters)

    def _read_frame_rows(self, component: str, timestamp: int, camera_name: int | None = None) -> pd.DataFrame:
        """Read only one frame from the local cache or GCS, without caching a full component."""
        filters = [("key.frame_timestamp_micros", "==", timestamp)]
        if camera_name is not None:
            filters.append(("key.camera_name", "==", camera_name))
        return self._read(component, filters=filters).compute()

    def _read_cached(self, component: str) -> pd.DataFrame:
        """Compute and cache a component DataFrame for the current segment.

        Repeated calls within the same segment return the cached copy, avoiding
        redundant GCS reads — important in notebook mode.
        """
        if component not in self._df_cache:
            self._df_cache[component] = self._read(component).compute()
        return self._df_cache[component]

    def _assert_segment(self):
        assert self.context_name, "Call assign_segment() before loading data."

    # -----------------------------------------------------------------------
    # Debugging helper
    # -----------------------------------------------------------------------

    def debug_columns(self, component: str):
        """Print actual Parquet column names for a component.

        Call this if you hit a KeyError to verify the column names on disk
        match what the code expects.
        """
        self._assert_segment()
        df = self._read_cached(component)
        print(f'\nColumns in {component!r}:')
        for col in df.columns:
            print(f'  {col}  ({df[col].dtype})')

    # -----------------------------------------------------------------------
    # Segment discovery
    # -----------------------------------------------------------------------

    def list_segments(self) -> list[str]:
        """Return sorted list of all context names available for the split."""
        filesystem = gcsfs.GCSFileSystem()
        pattern = f'{self.GCS_BUCKET}/{self.split}/camera_image/*.parquet'
        paths = filesystem.glob(pattern)
        return sorted(os.path.basename(path).replace('.parquet', '') for path in paths)

    def assign_segment(self, context_name: str):
        """Set the active segment; clears the DataFrame cache."""
        self.context_name = context_name
        self._df_cache = {}

    # -----------------------------------------------------------------------
    # Notebook mode — load_* helpers
    # -----------------------------------------------------------------------

    def get_timestamps(self, component: str = 'camera_image') -> list[int]:
        """Sorted timestamps from one component; reads only its timestamp column."""
        self._assert_segment()
        df = self._read(component, columns=["key.frame_timestamp_micros"]).compute()
        return sorted(df['key.frame_timestamp_micros'].unique().tolist())

    def load_camera_frame(
        self, timestamp: int, camera_name: int
    ) -> np.ndarray:
        """Decode and return a single camera frame as a BGR numpy array.

        Args:
            timestamp:   key.frame_timestamp_micros value.
            camera_name: Integer camera ID (1=FRONT ... 5=SIDE_RIGHT).

        Returns:
            (H, W, 3) uint8 BGR array.
        """
        self._assert_segment()
        df = self._read_frame_rows("camera_image", timestamp, camera_name)
        row = df.iloc[0]
        jpeg = row[f'{_C_IMG}.image']
        return cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )

    def load_camera_boxes(
        self, timestamp: int, camera_name: int
    ) -> pd.DataFrame:
        """Return camera-box rows for one (timestamp, camera) pair."""
        self._assert_segment()
        return self._read_frame_rows("camera_box", timestamp, camera_name).copy()

    def load_lidar_boxes(self, timestamp: int) -> pd.DataFrame:
        """Return LiDAR-box rows for one timestamp."""
        self._assert_segment()
        return self._read_frame_rows("lidar_box", timestamp).copy()

    def load_lidar_points(
        self, timestamp: int, top_lidar_only: bool = False,
    ) -> list[np.ndarray]:
        """Convert range images to point clouds for one timestamp.

        Returns:
            List of (N, 3) float32 arrays (one per LiDAR laser) in vehicle
            frame.
        """
        self._assert_segment()
        lidar_df = self._read_frame_rows("lidar", timestamp)
        cal_df = self._read_cached("lidar_calibration")

        group = lidar_df[lidar_df['key.frame_timestamp_micros'] == timestamp]
        if top_lidar_only:
            top_group = group[group['key.laser_name'] == 1]
            group = top_group if not top_group.empty else group
        points_list = []
        for _, row in group.iterrows():
            laser_name = row['key.laser_name']
            cal_rows = cal_df[cal_df['key.laser_name'] == laser_name]
            if cal_rows.empty:
                continue
            pts = self._range_image_to_points(row, cal_rows.iloc[0])
            points_list.append(pts)
        return points_list

    def load_lidar_points_xyzi(self, timestamp: int) -> list[np.ndarray]:
        """Like load_lidar_points() but returns (N, 4) arrays with intensity.

        The fourth column is the normalised intensity from range-image channel 1.
        Use this instead of load_lidar_points() when training PointPillars or
        any model that benefits from reflectance information.

        Args:
            timestamp: key.frame_timestamp_micros value.

        Returns:
            List of (N, 4) float32 arrays [x, y, z, intensity], one per laser.
        """
        self._assert_segment()
        lidar_df = self._read_frame_rows("lidar", timestamp)
        cal_df   = self._read_cached("lidar_calibration")

        group = lidar_df[lidar_df['key.frame_timestamp_micros'] == timestamp]
        points_list: list[np.ndarray] = []
        for _, row in group.iterrows():
            laser_name = row['key.laser_name']
            cal_rows   = cal_df[cal_df['key.laser_name'] == laser_name]
            if cal_rows.empty:
                continue
            pts = self._range_image_to_points_xyzi(row, cal_rows.iloc[0])
            points_list.append(pts)
        return points_list

    def load_camera_calibration(self, camera_name: int) -> pd.Series:
        """Return the calibration row for one camera (static across frames).

        Access fields directly:
            row[f'{_CAM_CAL}.intrinsic.f_u']
            row[f'{_CAM_CAL}.extrinsic.transform']  # list of 16 floats
        """
        self._assert_segment()
        df = self._read_cached('camera_calibration')
        return df[df['key.camera_name'] == camera_name].iloc[0]

    def load_all_boxes_df(self) -> pd.DataFrame:
        """Full lidar_box DataFrame for EDA (all timestamps in segment)."""
        self._assert_segment()
        return self._read_cached('lidar_box').copy()

    # -----------------------------------------------------------------------
    # Notebook mode — optional / challenge-specific components
    # -----------------------------------------------------------------------

    def load_camera_segmentation(
        self, timestamp: int, camera_name: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Decode the panoptic segmentation mask for one camera frame.

        Panoptic label encoding (COCO convention):
            panoptic_pixel_value = semantic_class_id * divisor + instance_id

        Decoding:
            semantic_mask = panoptic_label // divisor
            instance_mask = panoptic_label %  divisor   (0 = no instance)

        Note: camera_segmentation is only available for segments annotated
        for the 2-D Panoptic Video Segmentation challenge. Loading this
        component for an unannotated segment will raise a GCS 404 error.

        Call debug_columns('camera_segmentation') to inspect actual column
        names if a KeyError occurs.

        Args:
            timestamp:   key.frame_timestamp_micros value.
            camera_name: Integer camera ID (1=FRONT … 5=SIDE_RIGHT).

        Returns:
            semantic_mask: (H, W) uint16 — semantic class ID per pixel.
                           Use CAM_SEG_LABELS[id] for the human-readable name.
            instance_mask: (H, W) uint16 — instance ID per pixel
                           (0 = background / no tracked instance).
            divisor:       int — the panoptic_label_divisor used to encode.
        """
        self._assert_segment()
        df = self._read_cached('camera_segmentation')
        row = df[
            (df['key.frame_timestamp_micros'] == timestamp) &
            (df['key.camera_name'] == camera_name)
        ].iloc[0]

        divisor = int(row[f'{_CAM_SEG}.panoptic_label_divisor'])
        label_bytes = row[f'{_CAM_SEG}.panoptic_label']

        # Panoptic label is a PNG-encoded uint16 image.
        # cv2.IMREAD_UNCHANGED preserves the bit-depth (returns uint16).
        label = cv2.imdecode(
            np.frombuffer(label_bytes, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        if label is None:
            raise ValueError(
                'cv2.imdecode returned None for panoptic_label bytes. '
                'Run debug_columns("camera_segmentation") to inspect the '
                'raw column type — the data may use a different encoding.'
            )
        label = label.astype(np.uint32)
        return (
            (label // divisor).astype(np.uint16),
            (label % divisor).astype(np.uint16),
            divisor,
        )

    def load_lidar_segmentation(
        self, timestamp: int
    ) -> list[np.ndarray]:
        """Load per-point semantic class labels aligned with load_lidar_points().

        The returned list has one (N,) int32 array per LiDAR laser, in the
        same laser order as load_lidar_points(). N equals the number of valid
        first-return range-image pixels (range > 0) for that laser, so the
        shapes match element-wise.

        Assumed encoding: range_image_return1 has shape (H, W, 2) decoded as
        int32 — channel 0 = semantic class ID, channel 1 = instance ID.

        Note: lidar_segmentation is only available for segments annotated for
        the 3-D Semantic Segmentation challenge.

        Call debug_columns('lidar_segmentation') to verify column names.

        Args:
            timestamp: key.frame_timestamp_micros value.

        Returns:
            List of (N,) int32 arrays — LIDAR_SEG_LABELS[id] gives the class
            name. If segmentation is absent for a particular laser (rare),
            that entry is filled with zeros (TYPE_UNDEFINED).
        """
        self._assert_segment()
        seg_df    = self._read_cached('lidar_segmentation')
        lidar_df  = self._read_cached('lidar')
        cal_df    = self._read_cached('lidar_calibration')

        seg_group   = seg_df[seg_df['key.frame_timestamp_micros'] == timestamp]
        lidar_group = lidar_df[lidar_df['key.frame_timestamp_micros'] == timestamp]

        seg_list: list[np.ndarray] = []
        for _, lidar_row in lidar_group.iterrows():
            laser_name = lidar_row['key.laser_name']

            # Skip lasers without calibration (same guard as load_lidar_points)
            if cal_df[cal_df['key.laser_name'] == laser_name].empty:
                continue

            # Build the valid mask from the paired range image so array sizes match
            range_image = ToolKit._decode_range_image(lidar_row)
            valid_flat  = (range_image[..., 0] > 0).reshape(-1)
            n_valid = int(valid_flat.sum())

            seg_rows = seg_group[seg_group['key.laser_name'] == laser_name]
            if seg_rows.empty:
                # No segmentation for this laser — pad with TYPE_UNDEFINED (0)
                seg_list.append(np.zeros(n_valid, dtype=np.int32))
                continue

            seg_row = seg_rows.iloc[0]
            seg_values = seg_row[f'{_L_SEG}.range_image_return1.values']
            seg_shape  = seg_row[f'{_L_SEG}.range_image_return1.shape']

            # Seg values may be pre-decoded int32 ndarray or raw bytes —
            # handle both for forward-compatibility.
            if isinstance(seg_values, np.ndarray) and seg_values.dtype == np.int32:
                seg_arr = seg_values.reshape(seg_shape)
            else:
                seg_arr = np.frombuffer(seg_values, dtype=np.int32).reshape(seg_shape)
            semantic_flat = seg_arr[..., 0].reshape(-1)
            seg_list.append(semantic_flat[valid_flat].astype(np.int32))

        return seg_list

    def load_camera_keypoints(
        self, timestamp: int, camera_name: int
    ) -> pd.DataFrame:
        """Return all 2-D human keypoints for one (timestamp, camera) pair.

        Each row in the returned DataFrame represents one keypoint for one
        person. Links to the camera_box component via key.camera_object_id.

        Expected key columns (verify with debug_columns('camera_hkp')):
            key.camera_object_id
            [CameraHumanKeypointsComponent].keypoints.type
                 1=nose  2=left_eye  3=right_eye  4=left_ear  5=right_ear
                 6=left_shoulder  7=right_shoulder  8=left_elbow
                 9=right_elbow   10=left_wrist    11=right_wrist
                12=left_hip      13=right_hip     14=left_knee
                15=right_knee    16=left_ankle    17=right_ankle
            [CameraHumanKeypointsComponent].keypoints.keypoint_2d.location_px.x
            [CameraHumanKeypointsComponent].keypoints.keypoint_2d.location_px.y
            [CameraHumanKeypointsComponent].keypoints.keypoint_2d.visibility.is_occluded

        Note: camera_hkp is available only for Pose Estimation challenge
        segments. Returns an empty DataFrame for unannotated segments (the
        GCS file may not exist — catch gcsfs.exceptions.HTTPError if needed).

        Args:
            timestamp:   key.frame_timestamp_micros value.
            camera_name: Integer camera ID.

        Returns:
            pd.DataFrame — empty if no keypoints annotated for this frame.
        """
        self._assert_segment()
        df = self._read_cached('camera_hkp')
        return df[
            (df['key.frame_timestamp_micros'] == timestamp) &
            (df['key.camera_name'] == camera_name)
        ].copy()

    def load_projected_lidar_boxes(
        self, timestamp: int, camera_name: int
    ) -> pd.DataFrame:
        """Return 3-D LiDAR boxes projected into one camera image.

        The projected_lidar_box component provides 3-D LiDAR boxes
        reprojected into 2-D camera coordinates. Useful for cross-modal
        ground-truth alignment without manual projection math.

        Expected key columns (verify with debug_columns('projected_lidar_box')):
            key.laser_name          — source LiDAR sensor
            key.laser_object_id     — links to the lidar_box component
            [ProjectedLiDARBoxComponent].box.center.x   (pixels)
            [ProjectedLiDARBoxComponent].box.center.y   (pixels)
            [ProjectedLiDARBoxComponent].box.size.x     (pixels, width)
            [ProjectedLiDARBoxComponent].box.size.y     (pixels, height)
            [ProjectedLiDARBoxComponent].type            (int label type)

        Args:
            timestamp:   key.frame_timestamp_micros value.
            camera_name: Integer camera ID.

        Returns:
            pd.DataFrame with one row per projected 3-D box visible in this
            camera at this timestamp.
        """
        self._assert_segment()
        df = self._read_cached('projected_lidar_box')
        return df[
            (df['key.frame_timestamp_micros'] == timestamp) &
            (df['key.camera_name'] == camera_name)
        ].copy()

    def load_vehicle_pose(self, timestamp: int) -> np.ndarray:
        """Return the 4×4 world-from-vehicle transform for one frame.

        The transform maps points in vehicle frame (x=forward, y=left, z=up)
        to the world coordinate frame. The translation column T[:3, 3] gives
        the vehicle position in the world frame — useful for trajectory plots.

        Column accessed:
            [VehiclePoseComponent].world_from_vehicle.transform
            — list of 16 float64 values (row-major 4×4 matrix, same layout
              as the camera/lidar extrinsic transforms)

        Args:
            timestamp: key.frame_timestamp_micros value.

        Returns:
            (4, 4) float64 numpy array — world_from_vehicle transform.
        """
        self._assert_segment()
        df  = self._read_cached('vehicle_pose')
        row = df[df['key.frame_timestamp_micros'] == timestamp].iloc[0]
        return np.array(
            row[f'{_VEH_POSE}.world_from_vehicle.transform'],
            dtype=np.float64,
        ).reshape(4, 4)

    def load_all_vehicle_poses(self) -> list[tuple[int, np.ndarray]]:
        """Return [(timestamp, T_world_vehicle), ...] for all frames.

        Sorted by timestamp ascending so connecting positions in returned
        order traces the driven trajectory.

        Returns:
            List of (timestamp_micros, 4×4 float64 matrix) tuples.
        """
        self._assert_segment()
        df = self._read_cached('vehicle_pose')
        result: list[tuple[int, np.ndarray]] = []
        for _, row in df.iterrows():
            ts = int(row['key.frame_timestamp_micros'])
            T  = np.array(
                row[f'{_VEH_POSE}.world_from_vehicle.transform'],
                dtype=np.float64,
            ).reshape(4, 4)
            result.append((ts, T))
        return sorted(result, key=lambda x: x[0])

    def get_segment_stats(self) -> pd.DataFrame:
        """Return the full stats DataFrame for the current segment.

        The stats component contains per-frame difficulty metadata used for
        stratified evaluation across the Waymo challenges. Useful for:
          * Filtering frames by LEVEL_1 / LEVEL_2 difficulty
          * Verifying annotation coverage before training
          * Sampling balanced mini-datasets

        Call debug_columns('stats') to inspect available columns.

        Returns:
            pd.DataFrame with all stats rows for the segment.
        """
        self._assert_segment()
        return self._read_cached('stats').copy()

    # -----------------------------------------------------------------------
    # Extraction mode — write files to disk
    # -----------------------------------------------------------------------

    def extract_camera_images(self):
        """Decode every camera frame and write images + 2-D label files.

        Images   -> camera/images/<ts>_<cam>.png
        Labels   -> camera/labels/<ts>_<cam>.txt
                   (one line per box: type,x1,y1,w,h,object_id)
        """
        self._assert_segment()
        cam_image_df = self._read_cached('camera_image')
        cam_box_df = self._read_cached('camera_box')

        for _, img_row in cam_image_df.iterrows():
            ts = img_row['key.frame_timestamp_micros']
            cam_name_int = img_row['key.camera_name']
            cam_name = CAMERA_NAMES.get(cam_name_int, 'UNKNOWN')

            jpeg = img_row[f'{_C_IMG}.image']
            img = cv2.imdecode(
                np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            cv2.imwrite(
                f'{self.camera_images_dir}/{ts}_{cam_name}.png', img
            )

            mask = (
                (cam_box_df['key.frame_timestamp_micros'] == ts) &
                (cam_box_df['key.camera_name'] == cam_name_int)
            )
            with open(
                f'{self.camera_labels_dir}/{ts}_{cam_name}.txt', 'w'
            ) as f:
                for _, box_row in cam_box_df[mask].iterrows():
                    cx = box_row[f'{_C_BOX}.box.center.x']
                    cy = box_row[f'{_C_BOX}.box.center.y']
                    w = box_row[f'{_C_BOX}.box.size.x']
                    h = box_row[f'{_C_BOX}.box.size.y']
                    t = LABEL_TYPES.get(
                        int(box_row[f'{_C_BOX}.type']), 'TYPE_UNKNOWN'
                    )
                    oid = box_row['key.camera_object_id']
                    f.write(
                        f'{t},{cx - w/2:.2f},{cy - h/2:.2f},'
                        f'{w:.2f},{h:.2f},{oid}\n'
                    )

    def extract_lidar_labels(self):
        """Write 3-D box CSVs to lidar/labels/<ts>.csv.

        Columns: cx,cy,cz,sx,sy,sz,heading,type,object_id
        """
        self._assert_segment()
        lidar_box_df = self._read_cached('lidar_box')

        for ts, group in lidar_box_df.groupby('key.frame_timestamp_micros'):
            with open(f'{self.lidar_labels_dir}/{ts}.csv', 'w') as f:
                for _, row in group.iterrows():
                    t = LABEL_TYPES.get(
                        int(row[f'{_L_BOX}.type']), 'TYPE_UNKNOWN'
                    )
                    f.write(
                        f'{row[f"{_L_BOX}.box.center.x"]:.4f},'
                        f'{row[f"{_L_BOX}.box.center.y"]:.4f},'
                        f'{row[f"{_L_BOX}.box.center.z"]:.4f},'
                        f'{row[f"{_L_BOX}.box.size.x"]:.4f},'
                        f'{row[f"{_L_BOX}.box.size.y"]:.4f},'
                        f'{row[f"{_L_BOX}.box.size.z"]:.4f},'
                        f'{row[f"{_L_BOX}.box.heading"]:.6f},'
                        f'{t},{row["key.laser_object_id"]}\n'
                    )

    def extract_lidar_points(self):
        """Convert range images to point clouds and pickle them.

        Output: lidar/points/<ts>.pkl -- list of (N, 3) float32 arrays,
        one per LiDAR laser.
        """
        self._assert_segment()
        lidar_df = self._read_cached('lidar')
        cal_df = self._read_cached('lidar_calibration')

        for ts, group in lidar_df.groupby('key.frame_timestamp_micros'):
            points_per_laser = []
            for _, row in group.iterrows():
                laser_name = row['key.laser_name']
                cal_rows = cal_df[cal_df['key.laser_name'] == laser_name]
                if cal_rows.empty:
                    continue
                pts = self._range_image_to_points(row, cal_rows.iloc[0])
                points_per_laser.append(pts)
            with open(f'{self.lidar_points_dir}/{ts}.pkl', 'wb') as f:
                pickle.dump(points_per_laser, f)

    def export_yolo(
        self,
        output_dir: str,
        yolo_split: str = 'train',
        cameras: tuple[int, ...] = (1, 2, 3, 4, 5),
    ):
        """Export the segment in YOLO format for camera-based 2-D detection.

        Output layout (compatible with Ultralytics / YOLOX):
            <output_dir>/
              images/<yolo_split>/<context>_<ts>_<cam>.jpg
              labels/<yolo_split>/<context>_<ts>_<cam>.txt
              dataset.yaml                 ← written once; append-safe

        Label format (one line per box, values normalised 0–1):
            <class_id> <cx> <cy> <w> <h>

        Class mapping (TYPE_UNKNOWN is excluded):
            0 → vehicle
            1 → pedestrian
            2 → cyclist
            3 → sign

        Args:
            output_dir:  Root of the YOLO dataset on disk.
            yolo_split:  Subfolder name — 'train', 'val', or 'test'.
            cameras:     Camera IDs to export (default: all 5).
        """
        self._assert_segment()

        img_out_dir = os.path.join(output_dir, 'images', yolo_split)
        lbl_out_dir = os.path.join(output_dir, 'labels', yolo_split)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)

        cam_image_df = self._read_cached('camera_image')
        cam_box_df   = self._read_cached('camera_box')

        for _, img_row in cam_image_df.iterrows():
            cam_name_int = img_row['key.camera_name']
            if cam_name_int not in cameras:
                continue

            ts       = img_row['key.frame_timestamp_micros']
            cam_name = CAMERA_NAMES.get(cam_name_int, 'UNKNOWN')
            stem     = f'{self.context_name}_{ts}_{cam_name}'

            # Decode image and get dimensions
            jpeg = img_row[f'{_C_IMG}.image']
            img  = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8),
                                cv2.IMREAD_COLOR)
            h, w = img.shape[:2]

            # Save as JPEG (smaller than PNG; standard for YOLO datasets)
            cv2.imwrite(os.path.join(img_out_dir, f'{stem}.jpg'), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Write normalised label file
            mask = (
                (cam_box_df['key.frame_timestamp_micros'] == ts) &
                (cam_box_df['key.camera_name']            == cam_name_int)
            )
            boxes = cam_box_df[mask]

            with open(os.path.join(lbl_out_dir, f'{stem}.txt'), 'w') as f:
                for _, box_row in boxes.iterrows():
                    type_int = int(box_row[f'{_C_BOX}.type'])
                    if type_int not in YOLO_CLASS_MAP:
                        continue           # skip TYPE_UNKNOWN

                    cx = float(box_row[f'{_C_BOX}.box.center.x'])
                    cy = float(box_row[f'{_C_BOX}.box.center.y'])
                    bw = float(box_row[f'{_C_BOX}.box.size.x'])
                    bh = float(box_row[f'{_C_BOX}.box.size.y'])

                    # Normalise to [0, 1]
                    cx_n = cx / w
                    cy_n = cy / h
                    bw_n = bw / w
                    bh_n = bh / h

                    # Clamp to guard against boxes that slightly overflow
                    cx_n = min(max(cx_n, 0.0), 1.0)
                    cy_n = min(max(cy_n, 0.0), 1.0)
                    bw_n = min(bw_n, 1.0)
                    bh_n = min(bh_n, 1.0)

                    cls = YOLO_CLASS_MAP[type_int]
                    f.write(f'{cls} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}\n')

        # Write / update dataset.yaml (idempotent — safe to call per segment)
        yaml_path = os.path.join(output_dir, 'dataset.yaml')
        if not os.path.exists(yaml_path):
            with open(yaml_path, 'w') as f:
                f.write(f'path: {os.path.abspath(output_dir)}\n')
                f.write(f'train: images/train\n')
                f.write(f'val:   images/val\n')
                f.write(f'test:  images/test\n\n')
                f.write(f'nc: {len(YOLO_CLASS_NAMES)}\n')
                f.write(f'names: {YOLO_CLASS_NAMES}\n')

    # -----------------------------------------------------------------------
    # Range-image -> point-cloud conversion
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_beam_inclinations(cal_row: pd.Series, height: int) -> np.ndarray:
        """Return (height,) float32 beam-inclination angles for one laser."""
        incl_vals = cal_row[f'{_L_CAL}.beam_inclination.values']
        if (incl_vals is not None
                and hasattr(incl_vals, '__len__')
                and len(incl_vals) == height):
            return np.asarray(incl_vals, dtype=np.float32)
        incl_min = float(cal_row[f'{_L_CAL}.beam_inclination.min'])
        incl_max = float(cal_row[f'{_L_CAL}.beam_inclination.max'])
        return np.linspace(incl_max, incl_min, height, dtype=np.float32)

    @staticmethod
    def _decode_range_image(lidar_row: pd.Series) -> np.ndarray:
        """Return first-return range image as an (H, W, C) float32 ndarray."""
        ri_shape = lidar_row[f'{_L}.range_image_return1.shape']
        ri_values = lidar_row[f'{_L}.range_image_return1.values']
        return np.asarray(ri_values, dtype=np.float32).reshape(ri_shape)

    @staticmethod
    def _range_image_to_points(
        lidar_row: pd.Series,
        cal_row: pd.Series,
    ) -> np.ndarray:
        """Convert a range image to finite vehicle-frame (N, 3) XYZ points."""
        range_image = ToolKit._decode_range_image(lidar_row)
        ranges = range_image[..., 0]
        valid = np.isfinite(ranges) & (ranges > 0) & (ranges < 300.0)
        rows, cols = np.nonzero(valid)
        if not len(rows):
            return np.zeros((0, 3), dtype=np.float32)
        height, width, _ = range_image.shape
        inclination = ToolKit._get_beam_inclinations(cal_row, height)[rows]
        azimuth = np.linspace(np.pi, -np.pi, width, dtype=np.float32)[cols]
        selected_ranges = ranges[rows, cols]
        cos_inc = np.cos(inclination)
        xyz_sensor = np.stack([
            selected_ranges * cos_inc * np.cos(azimuth),
            selected_ranges * cos_inc * np.sin(azimuth),
            selected_ranges * np.sin(inclination),
        ], axis=1)
        extrinsic = np.asarray(cal_row[f'{_L_CAL}.extrinsic.transform'], dtype=np.float32).reshape(4, 4)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            xyz_vehicle = xyz_sensor @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        return xyz_vehicle[np.isfinite(xyz_vehicle).all(axis=1)].astype(np.float32, copy=False)


    @staticmethod
    def _range_image_to_points_xyzi(
        lidar_row: pd.Series,
        cal_row: pd.Series,
    ) -> np.ndarray:
        """Convert a range image to finite vehicle-frame (N, 4) XYZI points."""
        range_image = ToolKit._decode_range_image(lidar_row)
        ranges = range_image[..., 0]
        valid = np.isfinite(ranges) & (ranges > 0) & (ranges < 300.0)
        rows, cols = np.nonzero(valid)
        if not len(rows):
            return np.zeros((0, 4), dtype=np.float32)
        height, width, _ = range_image.shape
        inclination = ToolKit._get_beam_inclinations(cal_row, height)[rows]
        azimuth = np.linspace(np.pi, -np.pi, width, dtype=np.float32)[cols]
        selected_ranges = ranges[rows, cols]
        cos_inc = np.cos(inclination)
        xyz_sensor = np.stack([
            selected_ranges * cos_inc * np.cos(azimuth),
            selected_ranges * cos_inc * np.sin(azimuth),
            selected_ranges * np.sin(inclination),
        ], axis=1)
        extrinsic = np.asarray(cal_row[f'{_L_CAL}.extrinsic.transform'], dtype=np.float32).reshape(4, 4)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            xyz_vehicle = xyz_sensor @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        intensity = range_image[..., 1][rows, cols, None]
        output = np.concatenate((xyz_vehicle, intensity), axis=1)
        return output[np.isfinite(output).all(axis=1)].astype(np.float32, copy=False)
