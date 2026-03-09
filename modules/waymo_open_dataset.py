"""
waymo_open_dataset.py — Waymo Open Dataset v2 (Parquet format) toolkit.

All Parquet column names follow the official v2 naming convention:
  key fields      →  key.<field>
  component data  →  [ComponentClassName].<field>.<subfield>

No waymo-open-dataset pip package is required. Data is read with
dask/pandas/pyarrow and streamed directly from GCS.
TensorFlow is used only for tf.io.gfile (GCS listing) and range-image decode.

GCS layout:
  gs://waymo_open_dataset_v_2_0_0/<split>/<component>/<context_name>.parquet
"""

import os
import pickle

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import dask.dataframe as dd


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

    def __init__(self, split: str = 'training', save_dir: str = './output'):
        assert split in ('training', 'validation', 'testing'), (
            f"split must be 'training', 'validation', or 'testing',"
            f" got '{split}'"
        )
        self.split = split
        self.save_dir = save_dir
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
        return (f'gs://{self.GCS_BUCKET}/{self.split}'
                f'/{component}/{self.context_name}.parquet')

    def _read(self, component: str) -> dd.DataFrame:
        """Return a lazy Dask DataFrame for one component of the segment."""
        return dd.read_parquet(self._gcs_path(component))

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
        pattern = (
            f'gs://{self.GCS_BUCKET}/{self.split}/camera_image/*.parquet'
        )
        paths = tf.io.gfile.glob(pattern)
        return sorted(
            os.path.basename(p).replace('.parquet', '') for p in paths
        )

    def assign_segment(self, context_name: str):
        """Set the active segment; clears the DataFrame cache."""
        self.context_name = context_name
        self._df_cache = {}

    # -----------------------------------------------------------------------
    # Notebook mode — load_* helpers
    # -----------------------------------------------------------------------

    def get_timestamps(self) -> list[int]:
        """Sorted list of unique frame timestamps in the segment."""
        self._assert_segment()
        df = self._read_cached('camera_image')
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
        df = self._read_cached('camera_image')
        row = df[
            (df['key.frame_timestamp_micros'] == timestamp) &
            (df['key.camera_name'] == camera_name)
        ].iloc[0]
        jpeg = row[f'{_C_IMG}.image']
        return cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )

    def load_camera_boxes(
        self, timestamp: int, camera_name: int
    ) -> pd.DataFrame:
        """Return camera-box rows for one (timestamp, camera) pair."""
        self._assert_segment()
        df = self._read_cached('camera_box')
        return df[
            (df['key.frame_timestamp_micros'] == timestamp) &
            (df['key.camera_name'] == camera_name)
        ].copy()

    def load_lidar_boxes(self, timestamp: int) -> pd.DataFrame:
        """Return LiDAR-box rows for one timestamp."""
        self._assert_segment()
        df = self._read_cached('lidar_box')
        return df[df['key.frame_timestamp_micros'] == timestamp].copy()

    def load_lidar_points(self, timestamp: int) -> list[np.ndarray]:
        """Convert range images to point clouds for one timestamp.

        Returns:
            List of (N, 3) float32 arrays (one per LiDAR laser) in vehicle
            frame.
        """
        self._assert_segment()
        lidar_df = self._read_cached('lidar')
        cal_df = self._read_cached('lidar_calibration')

        group = lidar_df[lidar_df['key.frame_timestamp_micros'] == timestamp]
        points_list = []
        for _, row in group.iterrows():
            laser_name = row['key.laser_name']
            cal_rows = cal_df[cal_df['key.laser_name'] == laser_name]
            if cal_rows.empty:
                continue
            pts = self._range_image_to_points(row, cal_rows.iloc[0])
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
    def _range_image_to_points(
        lidar_row: pd.Series,
        cal_row: pd.Series,
    ) -> np.ndarray:
        """Convert one LiDAR's first-return range image to (N, 3) xyz.

        Args:
            lidar_row: Row from the 'lidar' component DataFrame.
            cal_row:   Matching row from the 'lidar_calibration' DataFrame.

        The standard spherical-to-Cartesian conversion is used:
            x = r * cos(inc) * cos(az)
            y = r * cos(inc) * sin(az)
            z = r * sin(inc)
        followed by the sensor extrinsic (sensor -> vehicle frame).
        """
        # --- decode range image ---
        ri_values = lidar_row[f'{_L}.range_image_return1.values']
        ri_shape = lidar_row[f'{_L}.range_image_return1.shape.dims']

        range_image = tf.reshape(
            tf.io.decode_raw(ri_values, tf.float32),
            ri_shape,
        )
        valid_mask = range_image[..., 0] > 0   # channel 0 = range in metres

        height, width = int(ri_shape[0]), int(ri_shape[1])

        # --- spherical coordinates ---
        azimuth = tf.cast(tf.linspace(np.pi, -np.pi, width), tf.float32)
        inclination = tf.cast(
            cal_row[f'{_L_CAL}.beam_inclinations'], tf.float32
        )

        inc_map = tf.broadcast_to(
            tf.reshape(inclination, [-1, 1]), [height, width]
        )
        az_map = tf.broadcast_to(azimuth, [height, width])

        r = range_image[..., 0]
        cos_inc = tf.cos(inc_map)
        x = r * cos_inc * tf.cos(az_map)
        y = r * cos_inc * tf.sin(az_map)
        z = r * tf.sin(inc_map)

        # --- apply sensor extrinsic (sensor -> vehicle frame) ---
        ones = tf.ones_like(x)
        xyz1 = tf.reshape(tf.stack([x, y, z, ones], axis=-1), [-1, 4])
        extrinsic = tf.cast(
            tf.reshape(cal_row[f'{_L_CAL}.extrinsic.transform'], [4, 4]),
            tf.float32,
        )
        xyz_vehicle = tf.matmul(xyz1, tf.transpose(extrinsic))[:, :3]

        valid_flat = tf.reshape(valid_mask, [-1])
        return tf.boolean_mask(xyz_vehicle, valid_flat).numpy()
