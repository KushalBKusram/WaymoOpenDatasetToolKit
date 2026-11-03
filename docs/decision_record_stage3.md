# Stage 3 Decision Record (draft)

Evidence: benchmark protocol v1.2, seed 42, 10 validation segments (1,967 FRONT
frames), single RTX 4080 SUPER. Error analysis from `scripts/error_analysis.py`.

## Decision

1. **YOLOv8n stays the camera baseline for the next stage.** RT-DETR-L is more
   accurate (mAP 0.120 vs 0.084; recall higher at every object size), and it
   clears the +0.02 mAP bar. It fails the protocol's second criterion: p95
   latency 32.7 ms is 3.1x YOLOv8n's 10.4 ms, above the allowed 2x. Under the
   pre-agreed rule the baseline stays, and RT-DETR-L is recorded as the
   accuracy-oriented alternative. The comparison is also confounded by size
   (RT-DETR-L has roughly ten times the parameters).
2. **Camera + LiDAR early fusion does not advance.** YOLOv8n with LiDAR depth and
   intensity channels scored mAP 0.081 vs 0.084 for camera-only, within noise,
   at about four times the latency.
3. **PointPillars and CenterPoint are tied.** BEV proxy scores were 0.088 and 0.102,
   but the official Waymo 3-D metric gives vehicle AP 0.048 for both, pedestrian
   0.013 and 0.014, cyclist 0 (Level 1, single seed). Neither advances over the
   other, and neither is comparable to camera mAP.

## Highest-impact failure modes

1. **Small objects.** 43% of ground-truth objects are smaller than 32x32 pixels in
   the original frame. YOLOv8n finds 4% of them and RT-DETR-L 12%. COCO mAP by
   size for YOLOv8n is 0.005 small, 0.093 medium, 0.327 large, so the overall
   0.084 mostly reflects small objects. Squashing 1920x1280 frames to 512x512
   shrinks them 2.5-3.75x before the network sees them.
2. **Loose boxes.** 65% (YOLOv8n) and 53% (RT-DETR-L) of false positives are
   localization errors: a box on a real object with IoU between 0.1 and 0.5.
   Another 31% / 37% are background boxes. Duplicates and class confusion are rare.
3. **Rare classes.** Cyclists (453 instances) are essentially never detected by
   any model; pedestrians are found only when medium or large.
4. **LiDAR 3-D accuracy.** Official vehicle AP (0.048) is about a quarter of the
   BEV proxy (about 0.2): the gap is probably height and heading error (not yet diagnosed).
   Pedestrian precision for both LiDAR models is about
   8% (tens of thousands of false positives).

## Next experiment

Raise the camera input resolution while keeping the aspect ratio (for example
960x640 or 1024x704, instead of 512x512 squashed) for YOLOv8n, and train on more
segments. Success criterion: small-object recall above 10% and mAP up by at least
0.02, with p95 latency still within 2x of the current YOLOv8n. If resolution
does not help, the limit is training data, and the next step is more segments.

## Result of the next experiment (2026-11-03)

Two runs of YOLOv8n at 640x960 (aspect ratio kept), same code and schedule apart from the data:

| Run | mAP | AP50 | Small / medium / large recall | Cyclist AP | p95 latency |
|---|---|---|---|---|---|
| Baseline, 512x512 squashed, 20 segments | 0.084 | 0.149 | 4.1% / 36% / 72% | 0.005 | 10.4 ms |
| A: 640x960, same 20 segments, 10 passes | 0.119 | 0.189 | 6.8% / 45% / 76% | 0.010 | 10.7 ms |
| B: 640x960, 100 segments, 3 passes | 0.157 | 0.256 | 11.5% / 51% / 79% | 0.118 | 11.1 ms |

- **Resolution alone** (A) adds +0.035 mAP at no latency cost.
- **More data** (B) adds a further +0.038 and makes cyclists detectable (medium recall 0% to 54%).
- B meets all three success criteria: small-object recall above 10%, mAP up by at least 0.02, latency
  within 2x. Under the protocol's advancement rule it replaces the 512x512 baseline for later stages.
- Costs: B produces more false positives (11.9k vs 6.3k, mostly background and loose boxes), and
  pedestrian AP dipped slightly (0.119 to 0.109), within what a second seed could change.
- Still true: single seed, the same 10 validation segments, FRONT camera only. B trained on 100 of
  798 segments for 3 passes, so it is a different budget from protocol v1.2 and needs its own protocol
  version before it is compared with other models.

## Caveats

Single seed; 20 of 798 training and 10 of 202 validation segments; the
LiDAR metric is a proxy; latency measured on one GPU type only.
