# Waymo Open Dataset ToolKit

A Python toolkit for exploring, visualising, and training on the
**Waymo Open Dataset v2** (Parquet format). Data streams directly from GCS
using `dask` / `pandas` / `pyarrow` — **no local download required** for EDA
or training. The official `waymo-open-dataset` pip package is **not** needed.

---

## Why this toolkit?

| Feature | This toolkit | Official devkit | mmdetection3d / OpenPCDet |
|---|:---:|:---:|:---:|
| GCS streaming (no download) | ✅ | ❌ | ❌ |
| All 12 v2 Parquet components | ✅ | ✅ | Partial |
| Notebook-ready `load_*` API | ✅ | ❌ | ❌ |
| YOLOv8 camera training on Colab | ✅ | ❌ | ❌ |
| PointPillars LiDAR training on Colab | ✅ | ❌ | ❌ |
| Switchable backbone registry | ✅ | ❌ | ❌ |
| `camera_hkp` keypoint loading | ✅ | ✅ | ❌ |
| macOS / Apple Silicon support | ✅ | ❌ | Partial |

---

## Quick start

### 1. Authenticate with Google Cloud

```bash
# Install gcloud CLI (once): https://cloud.google.com/sdk/docs/install
gcloud auth application-default login
```

This writes Application Default Credentials used automatically for all GCS reads.

### 2. Install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the local explorer (recommended)

```bash
streamlit run app.py
```

The app is an explorer and run-report dashboard only: training stays a
reliable, resumable CLI process. It first validates local Google Application
Default Credentials without contacting GCS, then lets you browse dataset
segments ten at a time and choose one focused workflow:

- **Camera Frames** — five synchronized, labeled camera views in a mosaic;
  open an individual view at full resolution when needed.
- **LiDAR Frames** — an interactive browser-based 3-D point cloud with a
  ground-reference surface, 3-D label boxes, hover details, frame navigation,
  and playback. A synchronized camera mosaic can be loaded for the active
  LiDAR timestamp.
- **Segment Analysis** — class balance, frame density, distance coverage, and
  per-class label-size summaries for the selected segment.

Selected workflow files are downloaded once into `.waymo_cache/` with per-file
progress, then reused locally. LiDAR exploration also persists its timestamp
index and decoded per-frame `.npz` cache. The **Local Cache** panel reports the
selected segment's storage use and can safely remove only that segment after a
confirmation step.

### 4. Open the EDA notebook (optional)

```bash
jupyter notebook notebooks/eda.ipynb
```

Data streams from GCS — you see annotated frames in seconds.

### Evaluation reports

Training writes `config.json` and `metrics.json` into `--drive-dir`; the
**Runs** page in the Streamlit app reads those artifacts. Evaluate a saved
checkpoint with:

```bash
python evaluate.py --config configs/yolov8n.yaml --run-dir ./runs/waymo
python evaluate.py --config configs/pointpillars.yaml --run-dir ./runs/waymo
```

Camera evaluation reports COCO-style 2-D mAP. The LiDAR report is a clearly
labelled BEV axis-aligned IoU proxy; use Waymo's official evaluator for
benchmark-quality rotated 3-D metrics.

---

## Project structure

```text
WaymoOpenDatasetToolKit/
├── configs/
│   ├── yolov8n.yaml            # YOLOv8-nano config (free Colab T4)
│   ├── yolov8s.yaml            # YOLOv8-small config (Colab Pro)
│   └── pointpillars.yaml       # PointPillars LiDAR detection (T4-friendly)
├── models/
│   ├── __init__.py             # Detector registry (register_detector / build_detector)
│   ├── base_detector.py        # Abstract BaseDetector interface
│   ├── yolov8_detector.py      # YOLOv8Detector implementation
│   └── pointpillars_detector.py # PointPillarsDetector implementation
├── modules/
│   ├── run_artifacts.py        # Training/evaluation JSON artifacts for the Runs page
│   ├── segment_cache.py        # On-demand local segment and decoded-frame cache
│   ├── waymo_open_dataset.py   # ToolKit class — GCS reader + all 12 components
│   └── visualize.py            # Visualisation utilities (camera, LiDAR, seg, poses)
├── notebooks/
│   ├── eda.ipynb               # Interactive EDA — 9 sections
│   ├── train.ipynb             # Colab training notebook — 2D camera detection
│   └── train_lidar.ipynb       # Colab training notebook — 3D LiDAR detection
├── train.py                    # Config-driven training script
├── evaluate.py                 # Checkpoint evaluation → JSON report
├── app.py                      # Streamlit local explorer + run dashboard
├── main.py                     # CLI entry point (argparse)
├── scripts/
│   └── download_data.sh        # Optional bulk GCS download helper
└── requirements.txt
```

---

## `modules/waymo_open_dataset.py` — `ToolKit`

Core class. Supports two usage modes:

| Mode | Description |
|---|---|
| **Notebook mode** | `load_*` methods return numpy arrays / DataFrames for interactive EDA |
| **Extraction mode** | `extract_*` methods write images, labels, and point clouds to disk |

### Segment management

```python
from modules.waymo_open_dataset import ToolKit

tk = ToolKit(split='training')          # 'training' | 'validation' | 'testing'
segments = tk.list_segments()           # all context names for the split
tk.assign_segment(segments[0])          # set active segment; clears cache
timestamps = tk.get_timestamps()        # sorted frame timestamps (µs)
```

### All 12 v2 Parquet components — `load_*` reference

#### Always-available components

| Method | Returns | Component |
|---|---|---|
| `load_camera_frame(ts, cam)` | `(H,W,3)` uint8 BGR array | `camera_image` |
| `load_camera_boxes(ts, cam)` | `pd.DataFrame` | `camera_box` |
| `load_lidar_points(ts)` | `list[(N,3) float32]` — one per laser | `lidar` |
| `load_lidar_boxes(ts)` | `pd.DataFrame` | `lidar_box` |
| `load_camera_calibration(cam)` | `pd.Series` — intrinsic + extrinsic | `camera_calibration` |
| `load_projected_lidar_boxes(ts, cam)` | `pd.DataFrame` — 3D boxes in image space | `projected_lidar_box` |
| `load_vehicle_pose(ts)` | `(4,4)` float64 — world_from_vehicle | `vehicle_pose` |
| `load_all_vehicle_poses()` | `list[(timestamp, 4×4)]` sorted by time | `vehicle_pose` |
| `get_segment_stats()` | `pd.DataFrame` — difficulty metadata | `stats` |
| `load_all_boxes_df()` | full `lidar_box` DataFrame (all frames) | `lidar_box` |

#### Challenge-specific components *(annotated segments only)*

| Method | Returns | Component | Challenge |
|---|---|---|---|
| `load_camera_segmentation(ts, cam)` | `(sem_mask, inst_mask, divisor)` | `camera_segmentation` | 2D Panoptic Seg |
| `load_lidar_segmentation(ts)` | `list[(N,) int32]` — per-point class IDs | `lidar_segmentation` | 3D Semantic Seg |
| `load_camera_keypoints(ts, cam)` | `pd.DataFrame` — 17-pt COCO skeleton rows | `camera_hkp` | Pose Estimation |

#### Batch / extraction methods

| Method | Output |
|---|---|
| `extract_camera_images()` | `camera/images/<ts>_<cam>.png` + `.txt` labels |
| `extract_lidar_labels()` | `lidar/labels/<ts>.csv` |
| `extract_lidar_points()` | `lidar/points/<ts>.pkl` — list of `(N,3)` arrays |
| `export_yolo(output_dir)` | YOLO-format images + labels + `dataset.yaml` |

### Debugging

```python
tk.debug_columns('camera_segmentation')   # print actual Parquet column names
```

Call this whenever you hit a `KeyError` — column names are printed with dtypes
so you can quickly spot mismatches.

### Column-name constants

```python
from modules.waymo_open_dataset import (
    _C_IMG, _C_BOX, _L_BOX, _L, _L_CAL, _CAM_CAL,   # core 6
    _CAM_SEG, _L_SEG, _CAM_HKP, _PROJ_BOX, _VEH_POSE, # new 5
)
```

### Label lookup tables

```python
from modules.waymo_open_dataset import (
    CAMERA_NAMES,           # {1: 'FRONT', 2: 'FRONT_LEFT', ...}
    LABEL_TYPES,            # {1: 'TYPE_VEHICLE', 2: 'TYPE_PEDESTRIAN', ...}
    LIDAR_SEG_LABELS,       # 23 semantic classes for lidar_segmentation
    CAM_SEG_LABELS,         # 29 semantic classes for camera_segmentation
    YOLO_CLASS_MAP,         # {1:0, 2:1, 4:2, 3:3} — Waymo type → YOLO class id
    YOLO_CLASS_NAMES,       # ['vehicle', 'pedestrian', 'cyclist', 'sign']
    LIDAR_DET_CLASS_MAP,    # {1:0, 2:1, 4:2} — Waymo type → LiDAR det class id
    LIDAR_DET_CLASS_NAMES,  # ['vehicle', 'pedestrian', 'cyclist']
)
```

---

## `modules/visualize.py` — Visualisation utilities

All `draw_*` functions return an annotated BGR numpy array.
All `plot_*` functions return a `matplotlib.Figure`.

### Functions

| Function | Description |
|---|---|
| `draw_camera_boxes(image, boxes_df)` | 2-D bounding box overlays on a camera frame |
| `draw_segmentation_mask(image, mask)` | Alpha-blend a 29-class panoptic mask |
| `draw_keypoints(image, keypoints_df)` | 17-point COCO skeleton on a camera frame |
| `draw_lidar_on_camera(image, points, calib)` | Depth-coloured LiDAR points projected onto camera |
| `plot_bev(points_list, boxes_df)` | Bird's-eye-view scatter coloured by height |
| `plot_ego_trajectory(poses)` | Plasma-coloured driven path in world coordinates |
| `build_open3d_scene(points_list, boxes_df)` | Open3D PointCloud + box wireframes (interactive) |
| `colorize_lidar_by_class(points, seg_labels)` | Open3D PointCloud coloured by semantic class |
| `project_lidar_to_camera(points, calib)` | Project vehicle-frame points into pixel space |

### Color maps

```python
from modules.visualize import (
    LABEL_COLORS_RGB,       # 5 object classes (vehicle, pedestrian, …)
    LIDAR_SEG_COLORS_RGB,   # 23 LiDAR semantic classes
    CAM_SEG_COLORS_RGB,     # 29 camera panoptic classes
)
```

### Camera IDs

| ID | Name |
|---|---|
| 1 | `FRONT` |
| 2 | `FRONT_LEFT` |
| 3 | `FRONT_RIGHT` |
| 4 | `SIDE_LEFT` |
| 5 | `SIDE_RIGHT` |

---

## EDA Notebook — `notebooks/eda.ipynb`

```bash
jupyter notebook notebooks/eda.ipynb
```

### Sections

| § | Title | What you see |
|---|---|---|
| 1 | Dataset Statistics | Object-class pie chart; 3-D boxes-per-frame histogram |
| 2 | Camera Frames | Single annotated frame; all-5-camera grid |
| 3 | LiDAR BEV | Bird's-eye-view scatter (height-coloured) + oriented 3-D box footprints |
| 4 | LiDAR 3-D (Open3D) | Interactive 3-D point cloud + box wireframes |
| 5 | LiDAR → Camera | Depth-coloured LiDAR overlay on the front camera |
| 6 | Camera Segmentation | Panoptic mask overlay + instance-ID heatmap *(challenge segs)* |
| 7 | LiDAR Semantic Seg | Side-by-side height-BEV vs class-BEV + class breakdown *(challenge segs)* |
| 8 | Human Keypoints | 17-pt COCO skeleton overlay on pedestrians *(challenge segs)* |
| 9 | Ego-Vehicle Trajectory | Full-segment driven path in world coordinates |

### Controls

```python
SPLIT     = 'training'   # 'training' | 'validation' | 'testing'
FRAME_IDX = 0            # index into the segment's timestamp list
CAMERA_ID = 1            # 1=FRONT  2=FRONT_LEFT  3=FRONT_RIGHT  4=SIDE_LEFT  5=SIDE_RIGHT
```

---

## Training — `notebooks/train.ipynb` + `train.py`

Trains **YOLOv8** for 2-D camera object detection on Waymo data streamed
directly from GCS. No local dataset export required.

### Running on Colab (free T4)

1. Open `notebooks/train.ipynb` in Google Colab
2. Set runtime to **T4 GPU** → *Runtime → Change runtime type*
3. Run all cells — the notebook handles GCS auth, repo clone, and dependency install

### Key configuration (notebook cell 2)

```python
CONFIG    = 'configs/yolov8n.yaml'          # nano — free T4
# CONFIG  = 'configs/yolov8s.yaml'          # small — Colab Pro
DRIVE_DIR = '/content/drive/MyDrive/waymo'  # checkpoint destination

# Optional one-off overrides (None = use the YAML value)
TOTAL_SEGS_OVERRIDE = None   # e.g. 5 for a quick smoke test
BATCH_OVERRIDE      = None   # e.g. 4 if you hit OOM
```

All other hyperparameters (lr, batch, epochs, img_size, eval thresholds) live
in the YAML. Edit the config file to make changes that persist across runs.

### Adding a new backbone

1. Create `models/<name>_detector.py` — subclass `BaseDetector`, decorate with `@register_detector('YourType')`
2. Add `from . import <name>_detector` to `models/__init__.py`
3. Create `configs/<name>.yaml` with `model.type: YourType`
4. Run: `python train.py --config configs/<name>.yaml`

No changes to `train.py` required.

### Resume after disconnect

Training auto-resumes from `DRIVE_DIR/checkpoints/latest.pt` — just re-run
the training cell. A `progress.json` file tracks which segments are done.

### Classes

| YOLO ID | Waymo type | Description |
|---|---|---|
| 0 | `TYPE_VEHICLE` | Cars, trucks, buses |
| 1 | `TYPE_PEDESTRIAN` | Pedestrians |
| 2 | `TYPE_CYCLIST` | Cyclists |
| 3 | `TYPE_SIGN` | Traffic signs |

### Evaluation (notebook section 6)

`torchmetrics.detection.MeanAveragePrecision` computes **mAP@0.5** per class
over configurable validation segments streamed live from GCS.

---

---

## LiDAR 3D Detection — `notebooks/train_lidar.ipynb` + `configs/pointpillars.yaml`

Trains **PointPillars** for 3-D LiDAR object detection on Waymo data streamed
directly from GCS. No local dataset export required.

### Architecture

| Stage | Module | Output shape |
|---|---|---|
| Voxelization | `_voxelize()` | `(P, 20, 9)` pillars |
| Pillar feature net | `PillarFeatureNet` (PointNet MLP + max-pool) | `(P, 64)` |
| Pillar scatter | `PillarScatter` (sparse → dense) | `(B, 64, 400, 400)` |
| BEV backbone | 2-stage 2-D CNN (64→128, stride 2) | `(B, 128, 200, 200)` |
| Detection head | heatmap + offset + Z + log-dim + heading | `(B, C+8, 200, 200)` |

**Detection range:** ±50 m XY, −3…3 m Z  
**Pillar size:** 0.25 m × 0.25 m → 400×400 BEV grid  
**Loss:** Gaussian focal loss (heatmap) + L1 (box regression)  
**Inference:** heatmap peak detection — no NMS required

### Running on Colab (free T4)

1. Open `notebooks/train_lidar.ipynb` in Google Colab
2. Set runtime to **T4 GPU** → *Runtime → Change runtime type*
3. Run all cells — notebook handles GCS auth, repo clone, and dependency install

### Key configuration (`configs/pointpillars.yaml`)

```yaml
voxel:
  x_range: [-50.0, 50.0]
  y_range: [-50.0, 50.0]
  z_range: [-3.0, 3.0]
  voxel_size: [0.25, 0.25]   # → 400×400 BEV
  max_pillars: 6000
  max_points_per_pillar: 20

train:
  batch_size: 2              # 2 × 6000 pillars fits ~5 GB VRAM
  lr: 1.0e-3
```

### Adding a new LiDAR backbone

1. Create `models/<name>_detector.py` — subclass `BaseDetector`, decorate with `@register_detector('YourType')`
2. Add `from . import <name>_detector` to `models/__init__.py`
3. Create `configs/<name>.yaml` with `task: lidar_3d_detection` and `model.type: YourType`
4. Run: `python train.py --config configs/<name>.yaml`

### LiDAR Classes

| Class ID | Waymo type | Description |
|---|---|---|
| 0 | `TYPE_VEHICLE` | Cars, trucks, buses |
| 1 | `TYPE_PEDESTRIAN` | Pedestrians |
| 2 | `TYPE_CYCLIST` | Cyclists |

### Evaluation (notebook section 7)

BEV mAP computed with axis-aligned IoU (fast proxy).
IoU thresholds follow Waymo convention: vehicle=0.7, pedestrian=0.5, cyclist=0.5.

> For competition-accurate evaluation use the official Waymo eval library
> (requires rotated BEV IoU). The notebook prints a note explaining the
> approximation.

---

## Waymo challenge coverage

| Challenge | Components used | Status |
|---|---|---|
| 2D Object Detection | `camera_image`, `camera_box` | ✅ Load + train (YOLOv8) |
| 3D Object Detection | `lidar`, `lidar_box`, `lidar_calibration` | ✅ Load + train (PointPillars) |
| 3D Object Tracking | `lidar_box` + `key.laser_object_id` | ✅ Data accessible |
| 2D Panoptic Video Seg | `camera_segmentation` | ✅ Load + visualise |
| 3D Semantic Segmentation | `lidar_segmentation` | ✅ Load + visualise |
| Occupancy & Flow | `lidar` (multi-frame) | 🔲 Data accessible |
| Motion Prediction | `vehicle_pose`, `lidar_box` | ✅ Pose loading + trajectory |
| Sim Agents | `vehicle_pose`, `lidar_box` | ✅ Data accessible |
| Pose Estimation | `camera_hkp` | ✅ Load + visualise |

---

## CLI — `main.py`

```bash
# List first 10 segments
python main.py --list

# Extract camera images + labels for one segment
python main.py --segment <context_name>

# Also extract LiDAR point clouds
python main.py --segment <context_name> --all

# Different split and output directory
python main.py --split validation --save-dir /tmp/out --segment <context_name>
```

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--split` | `training` | `training` / `validation` / `testing` |
| `--save-dir` | `./output` | Root directory for extracted files |
| `--list` | — | Print up to 10 segment names and exit |
| `--segment` | — | Context name to process |
| `--all` | — | Also extract LiDAR point cloud pickles |

**Extraction output layout:**

```text
<save_dir>/
├── camera/
│   ├── images/   <ts>_<CAMERA_NAME>.png
│   └── labels/   <ts>_<CAMERA_NAME>.txt    # type,x1,y1,w,h,object_id
└── lidar/
    ├── labels/   <ts>.csv                  # cx,cy,cz,sx,sy,sz,heading,type,object_id
    └── points/   <ts>.pkl                  # list of (N,3) float32 arrays
```

---

## Optional: bulk download

For offline / high-throughput workloads you can download segments locally:

```bash
./scripts/download_data.sh \
  waymo_open_dataset_v_2_0_0/training/camera_image \
  /mnt/data/waymo/training/camera_image \
  -m   # parallel download
```

Full dataset: ~2.29 TB. Check current size:
```bash
gsutil du -s -ah gs://waymo_open_dataset_v_2_0_0
```

---

## License

Licensed under [GNU AGPL v3](https://github.com/KushalBKusram/WaymoDataToolkit/blob/master/LICENSE).
