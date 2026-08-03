# Waymo Open Dataset Toolkit

Developer tooling for Waymo Open Dataset v2 Parquet data. The repository has
three independently usable surfaces:

- `app.py`: local Streamlit data explorer and run-artifact viewer.
- `train.py`: config-driven training for 2-D camera, camera–LiDAR fusion, and
  3-D LiDAR detectors.
- `evaluate.py`: checkpoint evaluation that persists a JSON report next to the
  training artifacts.

The data layer reads Waymo v2 Parquet components from GCS through
`dask`/`pyarrow`; it does not require the official TensorFlow-based
`waymo-open-dataset` package. Explorer data is cached deliberately in
`.waymo_cache`; CLI training streams the component rows it needs.

## Architecture

```text
Waymo v2 Parquet on GCS
        │
        ├── ToolKit (modules/waymo_open_dataset.py)
        │     ├── camera image / box samples
        │     ├── decoded LiDAR XYZI / 3-D boxes
        │     └── camera calibration
        │
        ├── Streamlit explorer (app.py)
        │
        └── train.py
              ├── detector registry (models/__init__.py)
              ├── camera datasets: RGB or RGB + projected LiDAR raster
              ├── LiDAR dataset: XYZI + 3-D boxes
              └── RunArtifacts: config.json, metrics.json, evaluation.json
```

A detector is selected by `model.type` in YAML. `train.py` dispatches the
input dataset from `task`; implementations follow `BaseDetector` for model
construction, batching, loss, and inference. Adding a detector should require
a model module, registry import, and configuration file—not a trainer fork.

## Environment and authentication

Python 3.11 is the recommended local runtime. Python 3.9 still runs the
current project but is end-of-life and produces dependency warnings.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GCS access uses Application Default Credentials (ADC). Install the Google Cloud
CLI if needed, then authenticate once:

```bash
gcloud auth application-default login
```

The Streamlit app checks for ADC before making GCS requests. Training or
segment listing fails if credentials are unavailable or cannot read the Waymo
bucket.

## Local explorer

```bash
streamlit run app.py
```

`app.py` is intentionally exploration/evaluation-only. It does not host a
long-running training job. The explorer:

- lists segments in pages of ten and downloads only selected components;
- renders synchronized five-camera mosaics;
- renders cached, interactive browser 3-D LiDAR point clouds and label boxes;
- exposes segment-level label, density, and distance statistics; and
- reads `RunArtifacts` JSON files for completed run reports.

The explorer owns `.waymo_cache/`. Deleting a segment through its cache control
removes only that local segment cache, never GCS data or training artifacts.

## Training

### Shipped backends

| `task` | `model.type` | Config | Input / output |
|---|---|---|---|
| `camera_2d_detection` | `YOLOv8Detector` | `configs/yolov8n.yaml`, `configs/yolov8s.yaml` | RGB camera image → 2-D boxes |
| `camera_2d_detection` | `TransformerDetector` | `configs/transformer.yaml` | RGB camera image → 2-D boxes |
| `camera_lidar_fusion_2d_detection` | `CameraLiDARFusionDetector` | `configs/camera_lidar_fusion.yaml` | RGB + calibrated LiDAR depth/intensity → 2-D boxes |
| `lidar_3d_detection` | `PointPillarsDetector` | `configs/pointpillars.yaml` | XYZI point cloud → oriented 3-D boxes |
| `lidar_3d_detection` | `CenterPointDetector` | `configs/centerpoint.yaml` | XYZI point cloud → oriented 3-D boxes |

The YOLO detector can use `n`, `s`, `m`, `l`, or `x` by changing
`model.backbone`. The Transformer and fusion detectors use a compact
CNN/Transformer with `hidden_dim`, `num_queries`, encoder/decoder layers, and
attention heads in their config. PointPillars and CenterPoint use the `voxel`
section for spatial bounds, pillar size, and point/pillar limits.

### Smoke tests

Run a bounded smoke test before a baseline. `--smoke-test` sets one segment,
one epoch, front camera where relevant, and ten timestamps. `--max-frames`
overrides the timestamp count for any backend.

```bash
# Validate the camera Transformer pipeline.
python train.py --config configs/transformer.yaml --smoke-test \
  --max-frames 1 --device mps --drive-dir ./runs/transformer-smoke

# Validate calibration, LiDAR decoding, and early-fusion training.
python train.py --config configs/camera_lidar_fusion.yaml --smoke-test \
  --max-frames 1 --device mps --drive-dir ./runs/fusion-smoke

# Validate full-resolution pillarization and 3-D loss.
python train.py --config configs/centerpoint.yaml --smoke-test \
  --max-frames 1 --device mps --drive-dir ./runs/centerpoint-smoke
```

A successful run produces:

```text
runs/<experiment>/
├── config.json                 # resolved configuration at run start
├── metrics.json                # append-only run/evaluation events
├── progress.json               # trained and pending segment names
├── evaluation.json             # written by evaluate.py
└── checkpoints/
    ├── latest.pt               # explicit resume target
    └── seg_XXXX.pt             # periodic checkpoints
```

A run directory containing `checkpoints/latest.pt` is protected. Use a new
`--drive-dir` for a new experiment; pass `--resume` only to continue that run.

```bash
python train.py --config configs/yolov8n.yaml \
  --drive-dir ./runs/yolov8n-baseline

python train.py --config configs/yolov8n.yaml \
  --drive-dir ./runs/yolov8n-baseline --resume
```

### Configuration and runtime controls

All config files share the following operational fields:

```yaml
name: experiment_name
task: camera_2d_detection

model:
  type: TransformerDetector

data:
  split: training
  cameras: [1, 2, 3, 4, 5]
  img_size: 512
  max_frames_per_seg: null

train:
  device: auto
  seed: 42
  total_segs: 20
  epochs_per_seg: 2
  batch_size: 4
  lr: 1.0e-4
  weight_decay: 1.0e-4
  grad_clip: 10.0
  save_every: 5
```

Use CLI flags for isolated experiments instead of modifying a tracked config:

```bash
python train.py --config configs/yolov8n.yaml \
  --total-segs 5 --max-frames 50 --batch 4 --lr 5e-5 --device mps
```

Runtime selection is explicit and shared by training and evaluation:

| Value | Selection |
|---|---|
| `auto` | CUDA, otherwise Apple Silicon MPS, otherwise CPU |
| `cuda` | CUDA-enabled PyTorch on an NVIDIA GPU |
| `mps` | Apple Silicon GPU via Metal Performance Shaders |
| `cpu` | CPU-only fallback |

Use CUDA for sustained multi-segment experiments. MPS is a valid local backend
for smoke tests and small pilots; it uses the Apple GPU, not the Neural Engine.

### Detector-specific implementation notes

**Transformer.** `TransformerDetector` is a compact DETR-style baseline with a
CNN feature encoder, 2-D positional encoding, learned object queries, and a
Transformer encoder/decoder. Its set matching is deterministic greedy matching,
not exact Hungarian matching. Treat it as an architecture comparison baseline,
not a reproduction of production DETR training.

**Camera–LiDAR fusion.** `CameraLiDARFusionDetector` projects vehicle-frame
XYZI through Waymo camera calibration and constructs a five-channel tensor:
RGB, nearest-return normalized depth, and intensity. It trains and evaluates
per camera (FRONT by default). This is calibrated early 2-D fusion, not
multi-camera fusion or 3-D fused detection.

**LiDAR.** PointPillars and CenterPoint consume the same XYZI and 3-D box
format. CenterPoint uses a separate multi-scale BEV backbone while retaining
the shared pillarizer, center targets, decoder, and evaluation interface,
allowing direct data/config comparisons.

## Evaluation

`evaluate.py` loads `checkpoints/latest.pt`, evaluates validation segments, and
writes `evaluation.json` and a metrics event into the supplied run directory.
Camera/fusion metrics use `torchmetrics` COCO mAP and require `pycocotools`
(already listed in `requirements.txt`).

```bash
python evaluate.py --config configs/yolov8n.yaml \
  --run-dir ./runs/yolov8n-baseline --max-frames 100

python evaluate.py --config configs/transformer.yaml \
  --run-dir ./runs/transformer-baseline --max-frames 100

# The fusion checkpoint must receive LiDAR and calibration for the same camera.
python evaluate.py --config configs/camera_lidar_fusion.yaml \
  --run-dir ./runs/fusion-baseline --cameras 1 --max-frames 100

python evaluate.py --config configs/centerpoint.yaml \
  --run-dir ./runs/centerpoint-baseline --max-frames 100
```

Camera and fusion reports are COCO-style 2-D mAP. LiDAR reports are explicitly
**BEV axis-aligned AP proxies**, not official Waymo rotated 3-D mAP. Do not use
them for benchmark or competition claims; integrating the official Waymo
evaluator is still required for that use case.

## Project structure

```text
WaymoOpenDatasetToolKit/
├── configs/
│   ├── yolov8n.yaml            # YOLOv8-nano config (free Colab T4)
│   ├── yolov8s.yaml            # YOLOv8-small config (Colab Pro)
│   ├── transformer.yaml        # Lightweight DETR-style camera detector
│   ├── camera_lidar_fusion.yaml # Calibrated RGB + LiDAR early fusion
│   ├── pointpillars.yaml       # PointPillars 3-D detection
│   └── centerpoint.yaml        # CenterPoint-style pillar 3-D detection
├── models/
│   ├── __init__.py             # Detector registry (register_detector / build_detector)
│   ├── base_detector.py        # Abstract BaseDetector interface
│   ├── yolov8_detector.py      # YOLOv8Detector implementation
│   ├── transformer_detector.py  # DETR-style query Transformer implementation
│   ├── fusion_detector.py       # Calibrated RGB + LiDAR detector
│   ├── pointpillars_detector.py # PointPillarsDetector implementation
│   └── centerpoint_detector.py  # CenterPoint-style LiDAR detector
├── modules/
│   ├── run_artifacts.py        # Training/evaluation JSON artifacts for the Runs page
│   ├── segment_cache.py        # On-demand local segment and decoded-frame cache
│   ├── fusion.py               # Calibration-aware LiDAR-to-camera rasterization
│   ├── runtime.py              # Device selection and reproducibility helpers
│   ├── waymo_open_dataset.py   # ToolKit class — GCS reader + all 12 components
│   └── visualize.py            # Visualisation utilities (camera, LiDAR, seg, poses)
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
| **In-memory mode** | `load_*` methods return numpy arrays / DataFrames for scripts and interactive inspection |
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

## Extending the detector registry

The trainer obtains a detector through `models.build_detector(cfg)`. A backend
must subclass `BaseDetector` and implement these operations:

| Method | Responsibility |
|---|---|
| `build_model(cfg, device)` | Construct/load the `nn.Module` and move it to the selected device. |
| `collate_fn(samples)` | Convert dataset samples to the batch representation used by the model. |
| `loss(model, batch)` | Return one scalar, gradient-bearing loss tensor. |
| `predict(model, input, cfg)` | Return detection dictionaries for evaluation. |

The camera sample contract is `(image, labels)`, where image is normalized
`(C, H, W)` and labels are `(N, 5)` in
`[class_id, center_x, center_y, width, height]` normalized coordinates. The
fusion backend changes image channels from RGB to RGB + depth + intensity. The
LiDAR sample contract is `(points, gt_boxes, gt_labels)`, with XYZI points and
boxes `[center_x, center_y, center_z, dx, dy, dz, heading]`.

To add a backend:

1. Add `models/<name>_detector.py`, subclass `BaseDetector`, and decorate the
   class with `@register_detector('Name')`.
2. Import the module from `models/__init__.py` so the decorator executes.
3. Add `configs/<name>.yaml` with the matching `model.type` and an existing or
   newly supported `task`.
4. Run a one-frame smoke test with a fresh `--drive-dir` before scaling it.

A new task type additionally needs a dataset-selection branch in `train.py`
and, if its inference input differs from RGB or XYZI, an evaluation branch in
`evaluate.py`. `CameraLiDARFusionDetector` is the reference for a calibrated
non-RGB camera task.

## Waymo challenge coverage

| Challenge | Components used | Status |
|---|---|---|
| 2D Object Detection | `camera_image`, `camera_box` | ✅ Load + train (YOLOv8, Transformer) |
| 3D Object Detection | `lidar`, `lidar_box`, `lidar_calibration` | ✅ Load + train (PointPillars, CenterPoint) |
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
