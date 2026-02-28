"""
WaymoOpenDataset.py — Waymo Open Dataset v2 (Parquet format) toolkit.

GCS layout:
  gs://waymo_open_dataset_v_2_0_0/<split>/<component>/<context_name>.parquet

Components used here:
  camera_image        — JPEG frames per camera per timestamp
  camera_box          — 2-D bounding boxes per camera frame
  camera_calibration  — Intrinsic + extrinsic per camera (static per segment)
  lidar_box           — 3-D bounding boxes per LiDAR sweep
  lidar               — Range images (return 1 & 2) per LiDAR per timestamp
  lidar_calibration   — Extrinsic + beam inclinations per LiDAR
"""

import os
import pickle

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2


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
    """

    GCS_BUCKET = 'waymo_open_dataset_v_2_0_0'

    def __init__(self, split: str = 'training', save_dir: str = './output'):
        """
        Args:
            split:    One of 'training', 'validation', 'testing'.
            save_dir: Root directory for extracted files (extraction mode only).
        """
        assert split in ('training', 'validation', 'testing'), \
            f"split must be 'training', 'validation', or 'testing', got '{split}'"

        self.split       = split
        self.save_dir    = save_dir
        self.context_name: str | None = None
        self._df_cache: dict[str, pd.DataFrame] = {}
        self._setup_dirs()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _setup_dirs(self):
        self.camera_images_dir = os.path.join(self.save_dir, 'camera', 'images')
        self.camera_labels_dir = os.path.join(self.save_dir, 'camera', 'labels')
        self.lidar_points_dir  = os.path.join(self.save_dir, 'lidar', 'points')
        self.lidar_labels_dir  = os.path.join(self.save_dir, 'lidar', 'labels')
        for d in (self.camera_images_dir, self.camera_labels_dir,
                  self.lidar_points_dir, self.lidar_labels_dir):
            os.makedirs(d, exist_ok=True)

    def _gcs_path(self, component: str) -> str:
        return (f'gs://{self.GCS_BUCKET}/{self.split}'
                f'/{component}/{self.context_name}.parquet')

    def _read(self, component: str) -> dd.DataFrame:
        """Return a lazy Dask DataFrame for one component of the active segment."""
        return dd.read_parquet(self._gcs_path(component))

    def _read_cached(self, component: str) -> pd.DataFrame:
        """Compute and in-memory cache a component DataFrame for the active segment.

        Repeated calls within the same segment return the cached copy, avoiding
        redundant GCS reads — important in notebook mode where multiple cells
        access the same component.
        """
        if component not in self._df_cache:
            self._df_cache[component] = self._read(component).compute()
        return self._df_cache[component]

    def _assert_segment(self):
        assert self.context_name, "Call assign_segment() before loading data."

    # -----------------------------------------------------------------------
    # Segment discovery
    # -----------------------------------------------------------------------

    def list_segments(self) -> list[str]:
        """Return a sorted list of all context names available for the split."""
        pattern = f'gs://{self.GCS_BUCKET}/{self.split}/camera_image/*.parquet'
        paths   = tf.io.gfile.glob(pattern)
        return sorted(os.path.basename(p).replace('.parquet', '') for p in paths)

    def assign_segment(self, context_name: str):
        """Set the active segment by its context name (filename without .parquet)."""
        self.context_name = context_name
        self._df_cache    = {}   # reset cache when the segment changes

    # -----------------------------------------------------------------------
    # Notebook mode — load_* helpers
    # -----------------------------------------------------------------------

    def get_timestamps(self) -> list[int]:
        """Return a sorted list of unique frame timestamps in the segment."""
        self._assert_segment()
        df = self._read_cached('camera_image')
        return sorted(df['key.frame_timestamp_micros'].unique().tolist())

    def load_camera_frame(self, timestamp: int, camera_name: int) -> np.ndarray:
        """Decode and return a single camera frame as a BGR numpy array.

        Args:
            timestamp:   key.frame_timestamp_micros value.
            camera_name: Integer camera ID (1=FRONT … 5=SIDE_RIGHT).

        Returns:
            (H, W, 3) uint8 BGR array.
        """
        self._assert_segment()
        df  = self._read_cached('camera_image')
        row = df[
            (df['key.frame_timestamp_micros'] == timestamp) &
            (df['key.camera_name'] == camera_name)
        ].iloc[0]
        cam = v2.CameraImageComponent.from_dict(row)
        return cv2.imdecode(np.frombuffer(cam.image, dtype=np.uint8), cv2.IMREAD_COLOR)

    def load_camera_boxes(self, timestamp: int, camera_name: int) -> pd.DataFrame:
        """Return the camera-box DataFrame rows for one frame + camera pair."""
        self._assert_segment()
        df = self._read_cached('camera_box')
        return df[
            (df['key.frame_timestamp_micros'] == timestamp) &
            (df['key.camera_name'] == camera_name)
        ].copy()

    def load_lidar_boxes(self, timestamp: int) -> pd.DataFrame:
        """Return the LiDAR-box DataFrame rows for one timestamp."""
        self._assert_segment()
        df = self._read_cached('lidar_box')
        return df[df['key.frame_timestamp_micros'] == timestamp].copy()

    def load_lidar_points(self, timestamp: int) -> list[np.ndarray]:
        """Convert range images to point clouds for one timestamp.

        Returns:
            List of (N, 3) float32 arrays (one per LiDAR laser) in vehicle frame.
        """
        self._assert_segment()
        lidar_df  = self._read_cached('lidar')
        cal_df    = self._read_cached('lidar_calibration')

        group = lidar_df[lidar_df['key.frame_timestamp_micros'] == timestamp]
        points_list = []
        for _, row in group.iterrows():
            laser_name = row['key.laser_name']
            cal_rows   = cal_df[cal_df['key.laser_name'] == laser_name]
            if cal_rows.empty:
                continue
            calibration = v2.LiDARCalibrationComponent.from_dict(cal_rows.iloc[0])
            lidar       = v2.LiDARComponent.from_dict(row)
            pts         = self._range_image_to_points(lidar, calibration)
            points_list.append(pts)
        return points_list

    def load_camera_calibration(self, camera_name: int) -> 'v2.CameraCalibrationComponent':
        """Return the calibration for one camera (static across frames)."""
        self._assert_segment()
        df  = self._read_cached('camera_calibration')
        row = df[df['key.camera_name'] == camera_name].iloc[0]
        return v2.CameraCalibrationComponent.from_dict(row)

    def load_all_boxes_df(self) -> pd.DataFrame:
        """Return the full lidar_box DataFrame for the segment (all timestamps).

        Useful for segment-level EDA (object counts, class distributions, etc.).
        """
        self._assert_segment()
        return self._read_cached('lidar_box').copy()

    # -----------------------------------------------------------------------
    # Extraction mode — write files to disk
    # -----------------------------------------------------------------------

    def extract_camera_images(self):
        """
        For every (timestamp, camera) pair in the segment:
          - decode the JPEG and write it to camera/images/<ts>_<cam>.png
          - write 2-D box labels to camera/labels/<ts>_<cam>.txt
            (one line per box: type,x1,y1,w,h,object_id)
        """
        self._assert_segment()

        cam_image_df = self._read_cached('camera_image')
        cam_box_df   = self._read_cached('camera_box')

        for _, img_row in cam_image_df.iterrows():
            ts           = img_row['key.frame_timestamp_micros']
            cam_name_int = img_row['key.camera_name']
            cam_name     = CAMERA_NAMES.get(cam_name_int, 'UNKNOWN')

            # --- image ---
            cam = v2.CameraImageComponent.from_dict(img_row)
            img = cv2.imdecode(
                np.frombuffer(cam.image, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            cv2.imwrite(f'{self.camera_images_dir}/{ts}_{cam_name}.png', img)

            # --- 2-D labels ---
            mask = (
                (cam_box_df['key.frame_timestamp_micros'] == ts) &
                (cam_box_df['key.camera_name'] == cam_name_int)
            )
            with open(f'{self.camera_labels_dir}/{ts}_{cam_name}.txt', 'w') as f:
                for _, box_row in cam_box_df[mask].iterrows():
                    box      = v2.CameraBoxComponent.from_dict(box_row)
                    cx, cy   = box.box.center.x, box.box.center.y
                    w, h     = box.box.size.x, box.box.size.y
                    x1, y1   = cx - w / 2, cy - h / 2
                    obj_type = LABEL_TYPES.get(box.type, 'TYPE_UNKNOWN')
                    f.write(
                        f'{obj_type},{x1:.2f},{y1:.2f},'
                        f'{w:.2f},{h:.2f},{box.key.camera_object_id}\n'
                    )

    def extract_lidar_labels(self):
        """
        For every timestamp write one CSV to lidar/labels/<ts>.csv.
        Columns: cx,cy,cz,sx,sy,sz,heading,type,object_id
        """
        self._assert_segment()
        lidar_box_df = self._read_cached('lidar_box')

        for ts, group in lidar_box_df.groupby('key.frame_timestamp_micros'):
            with open(f'{self.lidar_labels_dir}/{ts}.csv', 'w') as f:
                for _, row in group.iterrows():
                    box      = v2.LiDARBoxComponent.from_dict(row)
                    obj_type = LABEL_TYPES.get(box.type, 'TYPE_UNKNOWN')
                    f.write(
                        f'{box.box.center.x:.4f},{box.box.center.y:.4f},'
                        f'{box.box.center.z:.4f},{box.box.size.x:.4f},'
                        f'{box.box.size.y:.4f},{box.box.size.z:.4f},'
                        f'{box.box.heading:.6f},{obj_type},'
                        f'{row["key.laser_object_id"]}\n'
                    )

    def extract_lidar_points(self):
        """
        Convert range images to point clouds and pickle them to
        lidar/points/<ts>.pkl  as a list of (N, 3) float32 arrays,
        one per LiDAR laser (TOP, FRONT, SIDE_LEFT, SIDE_RIGHT, REAR).
        """
        self._assert_segment()
        lidar_df  = self._read_cached('lidar')
        lidar_cal_df = self._read_cached('lidar_calibration')

        for ts, group in lidar_df.groupby('key.frame_timestamp_micros'):
            points_per_laser = []
            for _, row in group.iterrows():
                laser_name = row['key.laser_name']
                cal_rows   = lidar_cal_df[lidar_cal_df['key.laser_name'] == laser_name]
                if cal_rows.empty:
                    continue
                calibration = v2.LiDARCalibrationComponent.from_dict(cal_rows.iloc[0])
                lidar       = v2.LiDARComponent.from_dict(row)
                points_per_laser.append(self._range_image_to_points(lidar, calibration))

            with open(f'{self.lidar_points_dir}/{ts}.pkl', 'wb') as f:
                pickle.dump(points_per_laser, f)

    # -----------------------------------------------------------------------
    # Range-image → point-cloud conversion (shared by both modes)
    # -----------------------------------------------------------------------

    @staticmethod
    def _range_image_to_points(
        lidar: 'v2.LiDARComponent',
        calibration: 'v2.LiDARCalibrationComponent',
    ) -> np.ndarray:
        """
        Convert a single LiDAR's first-return range image to a (N, 3) float32
        array of xyz points in the vehicle frame.

        The conversion uses spherical-to-Cartesian coordinates:
          x = r * cos(inclination) * cos(azimuth)
          y = r * cos(inclination) * sin(azimuth)
          z = r * sin(inclination)
        followed by the sensor extrinsic transform (sensor → vehicle frame).
        """
        ri = lidar.range_image_return1

        # Decode the compressed range image tensor — shape (H, W, 4)
        # Channel 0: range (metres), 1: intensity, 2: elongation, 3: is_in_nlz
        range_image = tf.reshape(
            tf.io.decode_raw(ri.values, tf.float32),
            ri.shape,
        )
        valid_mask = range_image[..., 0] > 0

        height, width = ri.shape[0], ri.shape[1]

        # Azimuth: uniform sweep from π → -π across the width
        azimuth = tf.cast(tf.linspace(np.pi, -np.pi, width), tf.float32)

        # Beam inclinations: one per row, provided by calibration
        inclination = tf.cast(calibration.beam_inclinations, tf.float32)

        inc_map = tf.broadcast_to(tf.reshape(inclination, [-1, 1]), [height, width])
        az_map  = tf.broadcast_to(azimuth, [height, width])

        r       = range_image[..., 0]
        cos_inc = tf.cos(inc_map)
        x = r * cos_inc * tf.cos(az_map)
        y = r * cos_inc * tf.sin(az_map)
        z = r * tf.sin(inc_map)

        # Build homogeneous coords (H*W, 4) then apply extrinsic
        ones       = tf.ones_like(x)
        xyz1       = tf.reshape(tf.stack([x, y, z, ones], axis=-1), [-1, 4])
        extrinsic  = tf.cast(
            tf.reshape(calibration.extrinsic.transform, [4, 4]),
            tf.float32,
        )
        xyz_vehicle = tf.matmul(xyz1, tf.transpose(extrinsic))[:, :3]

        # Discard invalid (zero-range) returns
        valid_flat = tf.reshape(valid_mask, [-1])
        return tf.boolean_mask(xyz_vehicle, valid_flat).numpy()
