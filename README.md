# Waymo Open Dataset Toolkit

## Description

A Python toolkit for exploring and extracting data from the **Waymo Open Dataset v2** (Parquet format). It streams data directly from GCS using `dask`/`pandas`/`pyarrow` — no local download required for EDA. The official `waymo-open-dataset` pip package is **not** needed.

---

## Getting Started

Ensure you have gained access to the dataset using your Google account. Proceed only after you can view the dataset on the [Google Cloud Storage browser](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_2_0_0).

### Install gcloud

- Follow the instructions on this [page](https://cloud.google.com/sdk/docs/install) to install the gcloud CLI.
- Authenticate with your account:

  ```bash
  gcloud auth application-default login
  ```

  This creates an Application Default Credentials file that is automatically used by the toolkit for GCS access.

### Install dependencies

```bash
pip install -r requirements.txt
```

> **macOS / Apple Silicon note:** `tensorflow` auto-selects the Metal build. The official `waymo-open-dataset-tf-*` package has Linux-only wheels and is intentionally excluded — all Parquet columns are accessed directly by their v2 names.

---

## Download Data (optional)

Streaming from GCS is sufficient for EDA. If you want a local copy, run:

```bash
./scripts/download_data.sh <source-blob> <destination-folder> [-m]
```

The `-m` flag enables parallel downloads. Example:

```bash
./scripts/download_data.sh \
  waymo_open_dataset_v_2_0_0/training/camera_image \
  /mnt/e/WaymoOpenDatasetV2/training/camera_image \
  -m
```

The full dataset is approximately **2.29 TB**. Check current size with:

```bash
gsutil du -s -ah gs://waymo_open_dataset_v_2_0_0
```

---

## Project Structure

```text
WaymoOpenDatasetToolKit/
├── main.py                      # CLI entry point (argparse)
├── modules/
│   ├── waymo_open_dataset.py    # ToolKit class — GCS reader + extractor
│   └── visualize.py             # Visualisation utilities
├── notebooks/
│   └── eda.ipynb                # Interactive EDA notebook
├── scripts/
│   └── download_data.sh         # Bulk GCS download helper
└── requirements.txt
```

---

## Modules

### `modules/waymo_open_dataset.py` — `ToolKit`

The core class. Supports two usage modes:

| Mode | Description |
|---|---|
| **Notebook mode** | `load_*` methods return in-memory numpy arrays / DataFrames for interactive EDA |
| **Extraction mode** | `extract_*` methods write images, labels, and point clouds to disk |

**Key methods:**

| Method | Description |
|---|---|
| `list_segments()` | Returns sorted list of all context names available for the split |
| `assign_segment(context_name)` | Sets the active segment; clears the DataFrame cache |
| `get_timestamps()` | Sorted list of unique frame timestamps in the active segment |
| `load_camera_frame(timestamp, camera_name)` | Returns a single camera frame as a BGR numpy array |
| `load_camera_boxes(timestamp, camera_name)` | Returns 2-D bounding box rows for one (timestamp, camera) pair |
| `load_lidar_boxes(timestamp)` | Returns 3-D LiDAR box rows for one timestamp |
| `load_lidar_points(timestamp)` | Converts range images to point clouds; returns list of `(N, 3)` arrays |
| `load_camera_calibration(camera_name)` | Returns the calibration row for one camera |
| `load_all_boxes_df()` | Full `lidar_box` DataFrame for EDA (all timestamps in segment) |
| `extract_camera_images()` | Decodes every camera frame and writes images + 2-D label `.txt` files |
| `extract_lidar_labels()` | Writes 3-D box CSVs per frame to `lidar/labels/` |
| `extract_lidar_points()` | Converts range images to point clouds and pickles them to `lidar/points/` |
| `debug_columns(component)` | Prints actual Parquet column names for a component (useful for debugging) |

**Output layout (extraction mode):**

```text
<save_dir>/
├── camera/
│   ├── images/   <timestamp>_<CAMERA_NAME>.png
│   └── labels/   <timestamp>_<CAMERA_NAME>.txt   # type,x1,y1,w,h,object_id
└── lidar/
    ├── labels/   <timestamp>.csv   # cx,cy,cz,sx,sy,sz,heading,type,object_id
    └── points/   <timestamp>.pkl   # list of (N, 3) float32 arrays
```

**Camera IDs:**

| ID | Name |
|---|---|
| 1 | FRONT |
| 2 | FRONT_LEFT |
| 3 | FRONT_RIGHT |
| 4 | SIDE_LEFT |
| 5 | SIDE_RIGHT |

---

### `modules/visualize.py` — Visualisation utilities

All public functions return annotated numpy arrays or matplotlib Figures, and display correctly in Jupyter without opening separate windows (except `build_open3d_scene` which opens an interactive viewer).

| Function | Description |
|---|---|
| `draw_camera_boxes(image, boxes_df)` | Overlays 2-D bounding boxes on a camera image |
| `plot_bev(points_list, boxes_df)` | Bird's-eye-view scatter of LiDAR points coloured by height, with oriented 3-D box footprints |
| `build_open3d_scene(points_list, boxes_df)` | Assembles an Open3D `PointCloud` + box `LineSet` wireframes for interactive 3-D viewing |
| `project_lidar_to_camera(points_list, cam_calib_row)` | Projects vehicle-frame LiDAR points into camera pixel coordinates |
| `draw_lidar_on_camera(image, points_list, cam_calib_row)` | Overlays depth-coloured LiDAR points onto a camera image |

---

## EDA Notebook

Open `notebooks/eda.ipynb` in Jupyter to explore a segment interactively. Data streams from GCS — no local download required.

```bash
jupyter notebook notebooks/eda.ipynb
```

### What the notebook covers

| Section | What you see |
|---|---|
| **1 — Dataset Statistics** | Object-class pie chart; 3-D boxes-per-frame histogram |
| **2 — Camera Frames** | Single frame with 2-D bounding box overlays; all-5-camera grid |
| **3 — LiDAR BEV** | Bird's-eye-view scatter (coloured by height) + oriented 3-D box footprints |
| **4 — LiDAR 3-D (Open3D)** | Interactive 3-D point cloud + box wireframes (opens separate window) |
| **5 — LiDAR → Camera** | Depth-coloured LiDAR points projected onto the front camera image |

### Key controls in the notebook

```python
SPLIT     = 'training'   # 'training' | 'validation' | 'testing'
SEGMENT   = segments[0]  # any context name returned by toolkit.list_segments()
FRAME_IDX = 0            # index into the segment's timestamp list
CAMERA_ID = 1            # 1=FRONT, 2=FRONT_LEFT, 3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT
```

---

## CLI Usage (`main.py`)

```bash
# List the first 10 available segments in the training split
python main.py --list

# List segments in a different split
python main.py --split validation --list

# Extract camera images + 2-D labels + 3-D LiDAR labels for one segment
python main.py --segment <context_name>

# Also extract LiDAR point cloud pickles (slower, more memory)
python main.py --segment <context_name> --all

# Use a different split and custom output directory
python main.py --split validation --save-dir /tmp/waymo_out \
               --segment <context_name>
```

**All CLI flags:**

| Flag | Default | Description |
| --- | --- | --- |
| `--split` | `training` | Dataset split: `training`, `validation`, or `testing` |
| `--save-dir` | `./output` | Root directory for extracted files |
| `--list` | — | Print up to 10 segment names and exit |
| `--segment` | — | Segment context name to process |
| `--all` | — | Also extract LiDAR point clouds |

---

## License

Licensed under [GNU AGPL v3](https://github.com/KushalBKusram/WaymoDataToolkit/blob/master/LICENSE).
