# Benchmark Protocol v1.3

This protocol defines how every model in the toolkit is trained and scored, so that differences between
results can be attributed to the model and not to the data, budget or metric. It covers camera,
camera + LiDAR fusion and LiDAR detection on Waymo Open Dataset v2. Earlier versions are summarised in
section 10; this document is the current, complete definition.

Two training settings are defined (section 1): **S** (small, the common budget for comparing different
models) and **L** (larger input and more data, run so far for YOLOv8n only). Every number in section 8 says
which setting it comes from.

## 1. Data splits and settings

- Training uses the Waymo v2 `training` split and evaluation uses `validation`. The splits are disjoint,
  so train and eval segments never overlap.
- **Evaluation set (all settings):** the first 10 sorted validation segments (of 202), 1,967 FRONT frames,
  frozen in `configs/benchmark_splits.json` (id `v1`).
- **Training sets:** frozen sorted prefixes of the training segments (of 798).

| Setting | Training segments | Manifest | Camera input | Schedule |
|---|---|---|---|---|
| **S** | first 20 (2.5%) | `benchmark_splits.json` (`v1`) | 512 x 512, squashed | 10 passes over 20 segments (about 40,000 image passes) |
| **S + resolution** (ablation) | first 20 | `benchmark_splits.json` (`v1`) | 640 x 960, aspect kept | 10 passes over 20 segments |
| **L** | first 100 (12.5%; contains the 20 of S) | `benchmark_splits_100.json` (`v1-train100`) | 640 x 960, aspect kept | 3 passes over 100 segments (about 59,000 image passes) |

S is deliberately cheap so a full benchmark is reproducible; it is a low-budget setting, not a statement
about what each model can do. The ablation isolates the effect of the input shape by keeping S's data and
schedule; L changes both input and data.

- Machines without Google Cloud credentials need the Parquet files of the segments in use cached locally
  (`.waymo_cache/`).

## 2. Training schedule

- Every frame of each training segment is used (about 198 per segment).
- Each visit to a segment trains `train.epochs_per_seg` = 1 epoch, and the segment order is reshuffled
  (seeded) every pass (`train.passes`). Earlier versions trained one segment for several epochs before moving
  on, which let the model forget earlier segments.
- Evaluation uses all frames of the 10 validation segments.
- `--max-frames` keeps a prefix of the first segment and is for smoke tests only. Never use it for reported
  numbers.
- Checkpoints are written every 5 segment visits and at the end of each pass; `--resume` restores model and
  optimizer state and continues the same pass.

## 3. Inputs, classes and cameras

- Camera models use the FRONT camera only. Setting S resizes (squashes) frames to 512 x 512 for training and
  evaluation; settings with a 640 x 960 input resize straight to that shape, which keeps the camera's 3:2
  aspect ratio. `data.img_size` accepts a single number (square) or `[height, width]`. Boxes are scaled back to
  the original frame.
- Camera classes: vehicle, pedestrian, cyclist, sign. The evaluated FRONT frames contain no ground-truth signs,
  so sign is effectively empty.
- LiDAR models use the TOP lidar only and three classes (vehicle, pedestrian, cyclist). They are not affected
  by the camera input shape and are run at setting S.
- Ground-truth boxes with fewer than 5 TOP-lidar points are excluded from LiDAR training and evaluation
  (`data.min_gt_points: 5`), as in Waymo's L1 setting.
- LiDAR frames are decoded once from range images (about 2 s each) and cached by
  `scripts/prepare_lidar_cache.py`. The cache must be rebuilt whenever the conversion changes.

## 4. Models

| Config | Model | Initialisation | Settings run |
|---|---|---|---|
| `yolov8n.yaml` | YOLOv8n, new 4-class head | COCO-pretrained backbone and box branches | S |
| `yolov8n_hires.yaml` | YOLOv8n at 640 x 960 | as YOLOv8n | S + resolution |
| `yolov8n_hires_100.yaml` | YOLOv8n at 640 x 960 | as YOLOv8n | L |
| `rtdetr.yaml` | RT-DETR-L, new 4-class head | COCO-pretrained | S |
| `yolo_fusion.yaml` | YOLOv8n with 5 input channels (RGB + projected LiDAR depth and intensity) | as YOLOv8n; extra channels start from the mean RGB filter | S |
| `pointpillars.yaml` | PointPillars-style center head | from scratch | S |
| `centerpoint.yaml` | CenterPoint-style pillar encoder + multi-scale BEV backbone | from scratch | S |

`transformer.yaml` and `camera_lidar_fusion.yaml` (a small DETR-style model) remain in the repository for
reference but are not part of the benchmark: at this budget they did not learn (mAP about 0.0002).

Size confound: RT-DETR-L is roughly ten times larger than YOLOv8n, so that comparison reflects model size as
well as architecture.

## 5. Metrics

- Camera and fusion: COCO mAP@[.50:.95] via torchmetrics, plus mAP50, mAP75 and per-class mAP.
  Operating-point precision and recall use confidence 0.25, IoU 0.5 and greedy per-class matching.
- LiDAR: **BEV axis-aligned AP proxy** (heading ignored), class IoU thresholds 0.7 / 0.5 / 0.5
  (vehicle / pedestrian / cyclist). Detections are local maxima of the heatmap above 0.2, at most 100 per class.
  This is not the official Waymo metric and must not be compared numerically with camera mAP.
- LiDAR official: Waymo 3-D detection AP and APH (Level 1 and 2) from `scripts/waymo_official_eval.py`, run
  on detections saved with `evaluate.py --dump-detections`. It needs Linux x86-64 with the
  `waymo-open-dataset-tf-2-12-0` wheel (Python 3.10, TensorFlow 2.13), so it runs apart from training.
  Difficulty is derived from point counts (5 or more points is Level 1) because the v2 validation labels carry
  none. Detections kept the 0.2 score cutoff, which slightly lowers AP. Sign has no model class and scores 0,
  so no mean over classes is reported.
- Latency: prediction-only mean, median and p95 from `evaluate.py`, measured with nothing else running on the
  GPU. It is only comparable on the same device. LiDAR latency (about 400 ms) is dominated by CPU voxelisation
  and decoding in Python, not the networks.
- Tracking metrics (MOTA, IDF1 and others) are defined in [tracking_baseline.md](tracking_baseline.md).

## 6. Seeds, naming and hardware

- Seed 42 for the reference runs. Differences smaller than the seed-to-seed spread must not be claimed as
  wins. The YOLOv8n runs at settings S and L were repeated with seeds 43 and 44 (section 8); other models have
  one seed.
- Run names: `<config>-v1.2-s<seed>` (for example `yolov8n-v1.2-s42`). The `v1.2` tag is the name the
  settings were first run under and is kept so existing run directories still match; v1.3 adds settings and
  metrics and does not change settings S. Runs from earlier protocol versions, and LiDAR runs made with the
  mis-registered point cloud, are kept under names that say so and are never reported.
- Reported runs used one NVIDIA RTX 4080 SUPER (32 GB). Latency and accuracy from different GPUs should not
  be mixed. (The reproducibility check re-evaluated a checkpoint on Apple MPS and matched to 0.0001 mAP; see
  [reproducibility.md](reproducibility.md).)

## 7. Criteria for advancing a model

A challenger replaces the baseline for the next stage if it improves the primary metric by at least 0.02 mAP
over YOLOv8n under this protocol, outside the seed-to-seed spread, with p95 latency no worse than 2x the
baseline's on the same device. Otherwise the baseline stays and the result is documented as a negative finding.

## 8. Reference results

To re-evaluate a reference run and check the numbers, see [reproducibility.md](reproducibility.md).
Scores are on the 10 evaluation segments; "seeds" is how many seeds were run. Per-class values are from seed 42.

| Model | Setting | Seeds | Score | Vehicle / Pedestrian / Cyclist | Latency (ms) |
|---|---|---|---|---|---|
| YOLOv8n | S | 3 | mAP 0.084 (0.082 to 0.087) | 0.185 / 0.062 / 0.005 | 10 |
| YOLOv8n | S + resolution | 1 | mAP 0.119 | 0.227 / 0.119 / 0.010 | 11 |
| **YOLOv8n** | **L** | 3 | **mAP 0.129 (0.108 to 0.157)** | 0.243 / 0.109 / 0.118 | 11 |
| RT-DETR-L | S | 1 | mAP 0.120 | 0.234 / 0.119 / 0.008 | 32 |
| YOLOv8n + LiDAR channels | S | 1 | mAP 0.081 | 0.172 / 0.071 / 0.000 | 44 |
| PointPillars | S | 1 | BEV proxy 0.088; official L1 AP 0.048 / 0.014 / 0 | proxy 0.196 / 0.067 / 0.000 | 405 |
| CenterPoint | S | 1 | BEV proxy 0.102; official L1 AP 0.048 / 0.013 / 0 | proxy 0.230 / 0.078 / 0.000 | 397 |

AP50 for seed 42: YOLOv8n S 0.149, S + resolution 0.189, L 0.256; RT-DETR-L 0.209; fusion 0.143. Camera
latencies are p95 for YOLOv8n and mean for the other camera rows (they differ by under 1 ms for YOLOv8n).

Seed repeats (YOLOv8n, same code, data and schedule):

| Setting | mAP seed 42 / 43 / 44 | Mean ± sd | Cyclist AP seed 42 / 43 / 44 |
|---|---|---|---|
| S | 0.0838 / 0.0823 / 0.0867 | 0.084 ± 0.002 | 0.005 / 0.000 / 0.000 |
| L | 0.1566 / 0.1078 / 0.1218 | 0.129 ± 0.025 | 0.118 / 0.007 / 0.000 |

Small-object recall (seed 42 only): 4.1% at S, 6.8% at S + resolution, 11.5% at L.

Reading these numbers:

- They are low-budget baselines. Camera training loss was still falling at the end of S; LiDAR loss had
  flattened.
- **Camera input and data.** Setting L beats S on every seed: mean gain +0.044 mAP, and the lowest L seed is
  +0.021 above the highest S seed, so the +0.02 bar holds on every pairing, narrowly in the worst. Vehicle and
  pedestrian AP are higher on all three seeds, with no latency cost. Seed 42 is the best L run: its cyclist AP
  and small-object recall did not repeat (cyclist AP near zero on the other seeds; small-object recall was not
  re-measured), so cyclist detection is not established. The seed spread is about ten times larger at L than at
  S (range 0.049 against 0.005 mAP), so differences under about 0.05 mAP at L should not be claimed. The
  resolution-only ablation has one seed, so the split between resolution (+0.035 on seed 42) and data (+0.038)
  is single-seed.
- **Model comparison at S.** RT-DETR-L (0.120) beats YOLOv8n (0.084) by more than the bar but at about 3x the
  latency, so under section 7 it does not advance. Adding LiDAR channels to YOLOv8n gave no measurable gain
  (0.081 against 0.084). RT-DETR-L, fusion and the LiDAR models have not been trained at L, so no cross-model
  claim is made at L.
- **LiDAR.** LiDAR scores are a different metric from camera mAP and cannot be ranked against them. Under the
  official metric (3-D, with heading) PointPillars and CenterPoint are tied (vehicle AP 0.048, pedestrian 0.013
  to 0.014, cyclist 0), so the proxy's ranking of CenterPoint ahead is not supported. The official AP is much
  lower than the proxy mainly because of height and horizontal localisation error (a local analysis found
  rotated 3-D recall 0.114 against 0.318 in rotated BEV for vehicles), with 22% of vehicles beyond the
  models' 50 m range; heading affects APH more than AP. See [findings.md](findings.md). Pedestrian precision for the
  LiDAR models is about 8%.

## 9. Known limitations

1. The BEV proxy ignores heading. The official evaluator is integrated for LiDAR only, on a single seed, with
   a 0.2 score cutoff.
2. Mostly single seed. Only YOLOv8n at S and L has a measured spread (0.005 and 0.049 mAP); other models and
   all LiDAR results have none.
3. Error slicing (object size, false-positive type) exists for camera runs only; there is no difficulty or
   distance slicing and LiDAR is not sliced.
4. Setting S uses 20 of 798 training segments and L uses 100; evaluation uses 10 of 202 validation segments.
5. The camera comparison between YOLOv8n and RT-DETR-L is confounded by model size.
6. Camera-only models use the FRONT view, and at setting S its 512 x 512 squash heavily downsamples small and
   distant objects.
7. Only the YOLOv8n camera model has been run at setting L.

## 10. Version history

- **v1 / v1.1:** the first frozen split and a schedule that trained one segment for several epochs before moving
  on. Results from these versions are not reported: the schedule caused forgetting, and a LiDAR point-cloud
  registration bug and a data-decoding bug invalidated early comparisons.
- **v1.2:** pass-based training over all segments, pretrained camera models, and the corrected data and LiDAR
  decoding. This is setting S.
- **v1.3:** adds the aspect-preserving camera input (`data.img_size: [height, width]`), the 100-segment
  setting L and the S + resolution ablation, the official LiDAR evaluator, camera error analysis, the tracking
  baseline, and seed repeats for YOLOv8n at S and L. Setting S is unchanged from v1.2.
