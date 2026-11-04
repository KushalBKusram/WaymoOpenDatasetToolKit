# Model card: YOLOv8n, 640 x 960, setting L (seed 42)

This model was made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset License
Agreement for Non-Commercial Use, available at [waymo.com/open/terms](https://waymo.com/open/terms), and your
access and use of such work are governed by the terms and conditions therein.

**Notice.** These weights are released under the Waymo Dataset License Agreement for Non-Commercial Use, and
any further downstream use or modification of them is subject to that agreement, including its
non-commercial restrictions (Section 4): they must not be used in the operation of a vehicle, in production
systems, or for primarily commercial purposes. A copy of the agreement is attached to the release as
`WAYMO_DATASET_LICENSE.txt`. The toolkit code is separately licensed under AGPL v3; this notice applies to the
weights. The model is also derived from Ultralytics YOLOv8 pretrained weights (AGPL-3.0).

## What it is

| | |
|---|---|
| Task | 2-D object detection on the Waymo FRONT camera |
| Architecture | YOLOv8n with a 4-class head (vehicle, pedestrian, cyclist, sign), COCO-pretrained backbone |
| Input | 640 x 960 (height x width), aspect ratio kept |
| File | `yolov8n_hires_100-v1.2-s42.weights.pt`, 12.7 MB, weights only (loads with `torch.load(..., weights_only=True)`) |
| SHA-256 | `af3cd3bea8803b68d8432049a207281d67642cad10a20ba55e988943a1e3ad5f` |
| Training | Setting L of the [benchmark protocol](benchmark_protocol.md): first 100 sorted Waymo v2 training segments, 3 passes, seed 42, config `configs/yolov8n_hires_100.yaml`, one RTX 4080 SUPER, Ultralytics 8.4.198 (as reported on the training machine) |

## Results

On the 10 frozen validation segments (1,967 frames), which the model was not trained on:

| | mAP@[.5:.95] | AP50 | Vehicle / Pedestrian / Cyclist AP |
|---|---|---|---|
| This model (seed 42) | 0.157 | 0.256 | 0.243 / 0.109 / 0.118 |
| Mean of 3 seeds (42, 43, 44) | 0.129 (range 0.108 to 0.157) | n/a | cyclist AP by seed 0.118 / 0.007 / 0.000 |

**Seed 42 is the best of the three seeds** at this setting, so expect about 0.13 mAP from a retrained model, and
do not rely on its cyclist detection, which did not repeat on the other seeds. Inference takes about 11 ms per
frame (p95) on the RTX 4080 SUPER. Full numbers and the comparison with other models are in the protocol.

## Intended use and limits

- Intended for research and benchmarking with this toolkit, on non-commercial terms.
- Trained on a small slice of the data (100 of 798 training segments), FRONT camera only. It misses most small
  and distant objects (the detector misses account for 90 to 97% of missed objects in the tracking analysis).
- Not validated for any safety-relevant use. It must not be used to operate or assist in operating a vehicle.

## Use

```python
import cv2, torch, yaml
from models import build_detector

cfg = yaml.safe_load(open("configs/yolov8n_hires_100.yaml"))
cfg["resume_weights"] = "yolov8n_hires_100-v1.2-s42.weights.pt"
detector = build_detector(cfg)
model = detector.build_model(cfg, torch.device("cpu")).eval()
for det in detector.predict(model, cv2.imread("frame.jpg"), cfg):      # BGR image
    print(det["box"], det["label"], det["score"])                       # xyxy in original pixels
```

To score it, put the file at `<run-dir>/checkpoints/latest.pt` and run `evaluate.py` as described in the
[reproducibility notes](reproducibility.md). Training checkpoints can be converted with
`scripts/export_weights.py`.

## Cite

```bibtex
@misc{waymo_open_dataset,
  title = {Waymo Open Dataset: An autonomous driving dataset},
  website = {\url{https://www.waymo.com/open}},
  year = {2019-2025}
}
```
