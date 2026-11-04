# Findings and next steps

What the benchmark showed, which models advance, what limits performance, and what to try next. The
numbers, settings (S and L) and metric definitions are in [benchmark_protocol.md](benchmark_protocol.md);
tracking results are in [tracking_baseline.md](tracking_baseline.md). Unless stated, results are seed 42 on
the 10 evaluation segments (1,967 FRONT frames) on one RTX 4080 SUPER; YOLOv8n at settings S and L also has
seeds 43 and 44.

## Decisions

1. **YOLOv8n is the camera baseline, and setting L replaces setting S for it.** RT-DETR-L is more accurate at
   setting S (mAP 0.120 against 0.084, higher recall at every object size) and clears the +0.02 bar, but its
   p95 latency of 32.7 ms is 3.1x YOLOv8n's 10.4 ms, above the allowed 2x. Under the pre-agreed rule the
   baseline stays and RT-DETR-L is recorded as the accuracy-oriented alternative; the comparison is also
   confounded by size (RT-DETR-L has about ten times the parameters). Separately, YOLOv8n at setting L
   (640 x 960, 100 segments) beats setting S on every seed: mean mAP 0.129 against 0.084, and the worst L seed
   is +0.021 above the best S seed, so the +0.02 bar holds on every pairing, narrowly in the worst, with no
   latency cost.
2. **Camera + LiDAR early fusion does not advance.** YOLOv8n with LiDAR depth and intensity channels scored
   mAP 0.081 against 0.084 camera-only, within noise, at about four times the latency.
3. **PointPillars and CenterPoint are tied.** The BEV proxy ranked CenterPoint ahead (0.102 against 0.088), but
   the official Waymo 3-D metric gives both vehicle AP 0.048, pedestrian 0.013 and 0.014, and cyclist 0
   (Level 1, single seed). Neither advances over the other, and neither is comparable to camera mAP.

## What limits performance

1. **Small objects.** 43% of ground-truth objects are smaller than 32 x 32 pixels in the original frame.
   At setting S, YOLOv8n finds 4% of them and RT-DETR-L 12%; COCO mAP by size for YOLOv8n is 0.005 small,
   0.093 medium and 0.327 large, so the overall 0.084 mostly reflects small objects. Squashing 1920 x 1280
   frames to 512 x 512 shrinks them 2.5 to 3.75 times before the network sees them. At setting L (seed 42)
   small-object recall rose to 11.5%; that was not re-measured on the other seeds.
2. **Loose boxes.** At setting S, 65% (YOLOv8n) and 53% (RT-DETR-L) of false positives are localization errors:
   a box on a real object with IoU between 0.1 and 0.5. Another 31% and 37% are background boxes. Duplicates
   and class confusion are rare. Setting L (seed 42) adds more false positives (11.9k against 6.3k), mostly
   background and loose boxes.
3. **Rare classes.** Cyclists (453 instances) are essentially never detected: setting S gives cyclist AP of
   0.005 or 0, and setting L gave 0.118 on seed 42 but 0.007 and 0.000 on the other seeds, so cyclist detection
   at L is not established. Pedestrians are found mainly when medium or large.
4. **LiDAR 3-D accuracy.** Official vehicle AP (0.048) is about a quarter of the BEV proxy (about 0.2).
   `scripts/lidar_error_analysis.py` (a local analysis of the saved detections, CenterPoint; PointPillars is
   similar) found:
   - Vehicle recall is 0.318 at rotated BEV IoU 0.7 but 0.114 at rotated 3-D IoU 0.7, so most hits are lost
     in 3-D. On matched pairs, using the ground-truth height and vertical size lifts the share reaching IoU
     0.7 from 0.21 to 0.58, and using the ground-truth horizontal centre lifts it to 0.45. Height (centre
     noise about 0.3 m) is the main loss and horizontal localisation second.
   - Heading is minor for AP: the median vehicle heading error is 3.6 degrees, and 18 to 20% of vehicle
     headings are flipped by 180 degrees, which affects APH and not AP.
   - 22% of official-L1 vehicles lie beyond the 50 m the models can see, which caps recall near 78%.
   - Precision is poor: about 31 vehicle predictions per frame against 20.5 ground-truth vehicles, and only
     34% of predictions scoring 0.7 or higher are true positives at IoU 0.7. Pedestrian precision is about 8%.
     Every cyclist prediction scores below 0.3, so cyclist AP is 0.
   - The dump and evaluator wiring look correct: an independent replica of the official matching gives vehicle
     AP 0.053 against 0.048 reported, and the replica reproduces the proxy exactly. The replica is not the
     official evaluator, and APH could not be checked. The cause of the height noise (pillar features have no
     vertical resolution) is an inference from the architecture, not tested. The 0.2 score cutoff should
     add at most about 0.001 to 0.003 AP by estimate.
5. **Tracking inherits the detector's misses.** For the camera tracker, 90 to 97% of missed objects are
   detector misses and most ID switches follow a gap in detection (details in the tracking doc).

## The experiment behind decision 1

After the setting S results, the next experiment was to keep the aspect ratio at a higher input (640 x 960
instead of a squashed 512 x 512) and train on more segments, with success meaning small-object recall above
10%, mAP up by at least 0.02, and p95 latency within 2x of YOLOv8n. On seed 42, resolution alone (same 20
segments) gave mAP 0.119 and 100 segments gave 0.157, with latency about 11 ms. The seed repeats put the
mean at 0.129 (range 0.108 to 0.157) and show that the 11.5% recall and the cyclist gain were specific to seed
42. The mAP and latency criteria hold on every seed; the recall criterion is met only where it was measured
(seed 42).

## Next experiments

- Repeat RT-DETR-L, fusion and the LiDAR models at setting L, and add seeds for the models that have one, so
  cross-model claims can be made at the larger budget.
- LiDAR: score the official metric on ground truth within 50 m for an in-range comparison, and retrain a
  LiDAR variant with extra loss weight on height (centre z and dz); more LiDAR training data is the other
  lever, since the loss had flattened.
- Tracking: shorten the confirmation delay for confident, large detections, and test appearance features to
  re-link tracks across gaps (the deferred ReID item).
- More training data, and class-balanced sampling, for cyclists.

## Caveats

Seeds 43 and 44 exist only for the two YOLOv8n camera settings; everything else is seed 42. Training uses 20
or 100 of 798 segments and evaluation 10 of 202. The LiDAR proxy metric is not the official one. Latency was
measured on one GPU type.
