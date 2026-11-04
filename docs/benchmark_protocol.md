# Benchmark Protocol v1.3

Status: v1.3. It adds a second, larger camera setting (section 10)
and the official LiDAR evaluator; the v1.2 setting below is unchanged and remains the common
budget for cross-model comparisons. v1.2 replaced the one-segment-at-a-time
schedule of v1/v1.1 with pass-based training, adds pretrained models, and reflects
the data and decoding bugs found while producing the first valid results.

Goal: every Stage 2 comparison (YOLOv8n vs RT-DETR, camera vs camera+LiDAR
fusion, PointPillars vs CenterPoint) uses the same data, budget, metric
definitions and hardware notes, so differences can be attributed to the model.

## 1. Data splits

- Training uses the Waymo v2 `training` split and evaluation uses `validation`.
  The splits are disjoint, so train and eval segments never overlap.
- The exact segments are frozen in `configs/benchmark_splits.json` (id `v1`):
  the first 20 sorted training segments (of 798) and the first 10 sorted
  validation segments (of 202). This is 2.5% and 5% of the data, chosen so a
  full benchmark is cheap and reproducible. It is a deliberate low-budget
  setting, not a statement about what each model can do.
- Machines without Google Cloud credentials need the 30 segments' Parquet
  files cached locally (`.waymo_cache/`).

## 2. Training schedule

- Every frame of each training segment is used (about 198 per segment).
- Training runs `train.passes` = 10 passes over all 20 segments. Each visit to
  a segment trains `epochs_per_seg` = 1 epoch, and the segment order is
  reshuffled (seeded) every pass. Earlier versions trained one segment for
  several epochs before moving on, which let the model forget earlier segments.
- Evaluation uses all frames of the 10 validation segments (1,967 frames).
- `--max-frames` keeps a prefix of the first segment and is for smoke tests
  only. Never use it for reported numbers.
- Checkpoints are written every 5 segment visits and at the end of each pass;
  `--resume` restores model and optimizer state and continues the same pass.

## 3. Inputs, classes and cameras

- Camera models use the FRONT camera only, resized (squashed) to 512 x 512 for
  both training and evaluation. Boxes are scaled back to the original frame.
  `data.img_size` also accepts `[height, width]` (for example `[640, 960]`, the
  camera's 3:2 aspect ratio, used in section 10); training and prediction then
  resize straight to that shape.
- Camera classes: vehicle, pedestrian, cyclist, sign. The evaluated FRONT
  frames contain no ground-truth signs, so sign is effectively empty.
- LiDAR models use the TOP lidar only and three classes (vehicle, pedestrian,
  cyclist).
- Ground-truth boxes with fewer than 5 TOP-lidar points are excluded from LiDAR
  training and evaluation (`data.min_gt_points: 5`), as in Waymo's L1 setting.
- LiDAR frames are decoded once from range images (about 2 s each) and cached
  by `scripts/prepare_lidar_cache.py`. The cache must be rebuilt whenever the
  conversion changes.

## 4. Models

| Config | Model | Initialisation |
|---|---|---|
| `yolov8n.yaml` | YOLOv8n, new 4-class head | COCO-pretrained backbone and box branches |
| `rtdetr.yaml` | RT-DETR-L, new 4-class head | COCO-pretrained |
| `yolov8n_hires.yaml`, `yolov8n_hires_100.yaml` | YOLOv8n at 640 x 960 (section 10) | as YOLOv8n |
| `yolo_fusion.yaml` | YOLOv8n with 5 input channels (RGB + projected LiDAR depth and intensity) | as YOLOv8n; extra channels start from the mean RGB filter |
| `pointpillars.yaml` | PointPillars-style center head | from scratch |
| `centerpoint.yaml` | CenterPoint-style pillar encoder + multi-scale BEV backbone | from scratch |

`transformer.yaml` and `camera_lidar_fusion.yaml` (a small DETR-style model)
remain in the repository for reference but are not part of the benchmark: at
this budget they did not learn (mAP about 0.0002).

Size confound: RT-DETR-L is roughly ten times larger than YOLOv8n, so that
comparison reflects model size as well as architecture.

## 5. Metrics

- Camera and fusion: COCO mAP@[.50:.95] via torchmetrics, plus mAP50, mAP75 and
  per-class mAP. Operating-point precision and recall use confidence 0.25,
  IoU 0.5 and greedy per-class matching.
- LiDAR: **BEV axis-aligned AP proxy** (heading ignored), class IoU thresholds
  0.7 / 0.5 / 0.5 (vehicle / pedestrian / cyclist). Detections are local maxima
  of the heatmap above 0.2, at most 100 per class. This is not the official
  Waymo metric and must not be compared numerically with camera mAP.
- LiDAR official: Waymo 3-D detection AP and APH (Level 1 and 2) from
  `scripts/waymo_official_eval.py`, run on detections saved with
  `evaluate.py --dump-detections`. It needs Linux x86-64 with the
  `waymo-open-dataset-tf-2-12-0` wheel (Python 3.10, TensorFlow 2.13), so it
  runs apart from training. Difficulty is derived from point counts (5 or more
  points is Level 1) because the v2 validation labels carry none. Detections
  kept the 0.2 score cutoff, which slightly lowers AP. Sign has no model class
  and scores 0, so no mean over classes is reported.
- Latency: prediction-only mean, median and p95 from `evaluate.py`, measured
  with nothing else running on the GPU. It is only comparable on the same
  device. LiDAR latency (about 400 ms) is dominated by CPU voxelisation and
  decoding in Python, not the networks.

## 6. Seeds, naming and hardware

- Seed 42 for the reference runs. Differences smaller than the seed-to-seed
  spread must not be claimed as wins. The two YOLOv8n camera settings (sections 8 and 10) were
  repeated with seeds 43 and 44: spread 0.005 mAP at 512 x 512 and 0.049 at 640 x 960. Other
  models have one seed.
- Run names: `<config>-v1.2-s<seed>` (for example `yolov8n-v1.2-s42`). Runs from
  earlier protocol versions, and LiDAR runs made with the mis-registered point
  cloud, are kept under names that say so and are never reported.
- Reported runs used one NVIDIA RTX 4080 SUPER (32 GB). Latency and accuracy
  from different GPUs should not be mixed.

## 7. Criteria for advancing a model

A challenger replaces the baseline for the next stage if it improves the primary
metric by at least 0.02 mAP over YOLOv8n under this protocol, outside the
seed-to-seed spread, with p95 latency no worse than 2x the baseline's on the
same device. Otherwise the baseline stays and the result is documented as a
negative finding.

## 8. Reference results (seed 42, single run)

To re-evaluate a reference run and check the numbers, see [reproducibility.md](reproducibility.md).

| Model | Score | AP50 | Vehicle / Pedestrian / Cyclist | Latency |
|---|---|---|---|---|
| YOLOv8n | mAP 0.084 | 0.149 | 0.185 / 0.062 / 0.005 | 9.7 ms |
| RT-DETR-L | mAP 0.120 | 0.209 | 0.234 / 0.119 / 0.008 | 32 ms |
| YOLOv8n + LiDAR channels | mAP 0.081 | 0.143 | 0.172 / 0.071 / 0.000 | 44 ms |
| PointPillars | BEV proxy 0.088; official L1 AP 0.048 / 0.014 / 0 | n/a | proxy 0.196 / 0.067 / 0.000 | 405 ms |
| CenterPoint | BEV proxy 0.102; official L1 AP 0.048 / 0.013 / 0 | n/a | proxy 0.230 / 0.078 / 0.000 | 397 ms |

Reading these numbers:

- They are low-budget baselines: 10 passes over about 4,000 training images.
  Camera training loss was still falling at the end; LiDAR loss had flattened.
- Adding LiDAR channels to YOLOv8n gave no measurable gain (0.081 vs 0.084).
- LiDAR scores are a different metric from camera mAP and cannot be ranked
  against them. Under the official metric (3-D, with heading) PointPillars and
  CenterPoint are tied (vehicle AP 0.048, pedestrian 0.013 to 0.014, cyclist 0),
  so the proxy's ranking of CenterPoint ahead is not supported.
- Pedestrian precision for the LiDAR models is about 8% and cyclists are
  essentially undetected by every model.

## 9. Known limitations

1. The BEV proxy ignores heading. The official evaluator is integrated for
   LiDAR only, on a single seed, with a 0.2 score cutoff.
2. Mostly single seed. For YOLOv8n the measured spread is 0.005 mAP (512 x 512) and 0.049 mAP
   (640 x 960, 100 segments); other models and the LiDAR results have no spread measured.
3. No difficulty, distance or size slicing of the errors yet.
4. The v1.2 setting uses only 20 of 798 training segments (100 in section 10) and 10 of 202
   validation segments.
5. The camera comparison between YOLOv8n and RT-DETR-L is confounded by model
   size.
6. Camera-only: FRONT view at 512 x 512, so small and distant objects are
   heavily downsampled.

## 10. Larger camera setting (v1.3)

A second budget, used to test whether the v1.2 camera numbers are limited by input
resolution or by training data. It is a different budget from the v1.2 table and must
not be mixed with it except as an explicit comparison.

- Input: 640 x 960 (height x width), aspect ratio kept, FRONT camera.
- Training segments: the first 100 sorted training segments (`configs/benchmark_splits_100.json`,
  id `v1-train100`; it contains the 20 v1.2 segments). Evaluation segments are identical to v1.
- Schedule: 3 passes of 1 epoch over the 100 segments (about 59,000 image passes), versus
  10 passes over 20 segments (about 40,000) for v1.2. Run `yolov8n_hires.yaml` isolates the
  resolution change by keeping the v1.2 data and schedule.
- Metrics, evaluation set and latency method are unchanged.

| Run (seed 42) | mAP | AP50 | Vehicle / Pedestrian / Cyclist | Small-object recall | p95 latency |
|---|---|---|---|---|---|
| v1.2 YOLOv8n, 512 x 512 | 0.084 | 0.149 | 0.185 / 0.062 / 0.005 | 4.1% | 10.4 ms |
| 640 x 960, 20 segments, 10 passes | 0.119 | 0.189 | 0.227 / 0.119 / 0.010 | 6.8% | 10.7 ms |
| 640 x 960, 100 segments, 3 passes | 0.157 | 0.256 | 0.243 / 0.109 / 0.118 | 11.5% | 11.1 ms |

Reading (seed 42): resolution alone adds +0.035 mAP and more data a further +0.038, without extra
latency. These are single-seed figures; see the repeats below. RT-DETR-L and the fusion and LiDAR
models have not been trained at this budget, so no cross-model claim is made here.

Seed repeats (seeds 42 / 43 / 44, same code, data and schedule):

| Setting | mAP per seed | Mean ± sd | Cyclist AP per seed |
|---|---|---|---|
| 512 x 512, 20 segments | 0.0838 / 0.0823 / 0.0867 | 0.084 ± 0.002 | 0.005 / 0.000 / 0.000 |
| 640 x 960, 100 segments | 0.1566 / 0.1078 / 0.1218 | 0.129 ± 0.025 | 0.118 / 0.007 / 0.000 |

The mean gain from the v1.3 setting is +0.044 mAP, and the lowest 640 x 960 seed is still +0.021 above
the highest 512 x 512 seed, so the +0.02 bar holds on every pairing, narrowly in the worst. Seed 42 is the
best 640 x 960 run, and its cyclist AP and small-object recall (11.5%) did not repeat: cyclist AP was
near zero on the other seeds, and small-object recall was not re-measured. Run `yolov8n_hires.yaml`
(resolution only, 20 segments) has one seed, so the split between resolution and data is single-seed.
