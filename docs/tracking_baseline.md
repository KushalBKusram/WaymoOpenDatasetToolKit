# Camera tracking baseline (Stage 4)

Status: draft. Single seed, single detector checkpoint (`yolov8n-v1.2-s42`), FRONT camera only.

## Method

- **Detector:** YOLOv8n from the benchmark (protocol v1.2). Detections are cached per frame in the
  run directory, so tracker settings can change without re-running the model.
- **Tracker:** `modules/tracking.py`, SORT-style. A constant-velocity Kalman filter per track over
  (cx, cy, w, h), Hungarian matching on IoU, matching only within a class. No appearance features.
- **Ground truth:** FRONT camera boxes with their persistent `camera_object_id`.
- **Scoring:** `scripts/track_eval.py` with `motmetrics`. A track matches a ground-truth box at
  IoU >= 0.5. Metrics are pooled per class across segments (identities never cross segments), and the
  headline MOTA and IDF1 weight each class by its number of ground-truth objects.
- **Evaluation data:** the same 10 frozen validation segments as the detection benchmark (1,967 frames).

## Checks on the pipeline

- **Identity oracle** (every ground-truth box gets its true id as a track) scores exactly 1.0 on every
  metric, so the scoring code is correct.
- **Box oracle** (ground-truth boxes as detections, through the tracker) scores MOTA 0.82 and IDF1 0.68.
  This is the ceiling of an IoU tracker on this data. Boxes of the same object overlap only about
  0.48 IoU (median) between consecutive frames on the segment checked, because many are small, and
  ground-truth labels disappear for up to 37 frames, longer than the default `max_age` of 5.

## Results (10 eval segments)

| Setup | MOTA | IDF1 | Vehicle recall / precision | Vehicle ID switches |
|---|---|---|---|---|
| YOLOv8n, default tracker (score 0.25, IoU 0.3, max_age 5, min_hits 2) | 0.105 | 0.286 | 0.31 / 0.64 | 516 |
| RT-DETR-L, same default tracker | -0.042 | 0.272 | 0.42 / 0.49 | 1,205 |
| **YOLOv8n, tuned tracker** (score 0.5, IoU 0.2, max_age 40, min_hits 1) | **0.166** | **0.307** | 0.28 / 0.79 | 328 |
| Box oracle (ceiling) | 0.819 | 0.679 | 0.90 / 1.00 | 1,616 |

Tuned settings were chosen on 3 held-out validation segments (`scripts/tune_tracker.py`, 144 settings,
about 590 frames, rule fixed in advance: best weighted IDF1, ties by MOTA) and then scored once on the
10 reported segments. On the dev segments the gain was IDF1 0.243 to 0.262 and MOTA 0.091 to 0.138, and the
top five settings were within 0.001 IDF1 of each other, so the optimum is flat.

## What this shows

1. **Detection is the main limit.** Even the tuned tracker loses most objects: vehicle recall is 0.28,
   cyclists are never tracked, and pedestrians stay at IDF1 0.17.
2. **Most of the gain came from the stricter score gate** (fewer false tracks), plus a longer track
   memory that reduces vehicle ID switches (516 to 328). Pedestrian switches rose (25 to 55) with the
   longer memory.
3. **A better detector alone does not fix tracking.** RT-DETR-L finds more objects but adds false
   positives at the same gate and switches identities more often. With a higher score gate it
   slightly beats YOLOv8n (IDF1 0.333 vs 0.289 at gate 0.6), but that sweep used the eval segments
   and is a sensitivity check only, not a result.
4. **The IoU tracker has a low ceiling.** With perfect boxes IDF1 is 0.68, so association needs
   appearance or motion cues for the remaining gap.

## Limits

Single seed and detector checkpoint; 3 tuning segments; only the FRONT camera; no HOTA (not in
`motmetrics`); the +0.02 IDF1 gain is within what a second seed could change and has not been repeated.

## Reproduce

```bash
python scripts/track_eval.py --config configs/yolov8n.yaml --run-dir runs/yolov8n-v1.2-s42
python scripts/tune_tracker.py --config configs/yolov8n.yaml --run-dir runs/yolov8n-v1.2-s42
python scripts/track_eval.py --config configs/yolov8n.yaml --run-dir runs/yolov8n-v1.2-s42 \
    --score-thresh 0.5 --iou-threshold 0.2 --max-age 40 --min-hits 1 --out runs/yolov8n-v1.2-s42/tracking/tracking_tuned.json
```
