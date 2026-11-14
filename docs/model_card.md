# Model card: YOLOv8n and RT-DETR-L, 640 x 960, setting L (seed 42)

These models were made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset License
Agreement for Non-Commercial Use, available at [waymo.com/open/terms](https://waymo.com/open/terms), and your
access and use of such work are governed by the terms and conditions therein.

**Notice.** These weights are released under the Waymo Dataset License Agreement for Non-Commercial Use, and
any further downstream use or modification of them is subject to that agreement, including its
non-commercial restrictions (Section 4): they must not be used in the operation of a vehicle, in production
systems, or for primarily commercial purposes. A copy of the agreement is attached to the release as
`WAYMO_DATASET_LICENSE.txt`. The toolkit code is separately licensed under AGPL v3; this notice applies to the
weights. Both models are derived from Ultralytics pretrained weights (AGPL-3.0).

Both files are weights only: they load with `torch.load(..., weights_only=True)` and contain no pickled code.
Both were trained on setting L of the [benchmark protocol](benchmark_protocol.md): the first 100 sorted Waymo
v2 training segments, 3 passes over them, 640 x 960 FRONT-camera input (aspect ratio kept), seed 42, one RTX
4080 SUPER, Ultralytics 8.4.198 (as reported on the training machine). Four classes: vehicle, pedestrian,
cyclist, sign (sign is empty in the evaluated frames).

| | YOLOv8n | RT-DETR-L |
|---|---|---|
| File | `yolov8n_hires_100-v1.2-s42.weights.pt` (12.7 MB) | `rtdetr_l_hires_100-v1.2-s42.weights.pt` (131.8 MB) |
| SHA-256 | `af3cd3bea8803b68d8432049a207281d67642cad10a20ba55e988943a1e3ad5f` | `ea1283784d327bc833f287c45d1aa92bbdfc43f4fe2bd38e380410085ee38451` |
| Config | `configs/yolov8n_hires_100.yaml` | `configs/rtdetr_hires_100.yaml` |
| Initialisation | COCO-pretrained backbone, new 4-class head | COCO-pretrained, new 4-class head |

## Results

On the 10 frozen validation segments (1,967 frames), which the models were not trained on:

| | mAP@[.5:.95] | AP50 | Vehicle / Pedestrian / Cyclist AP | p95 latency |
|---|---|---|---|---|
| YOLOv8n, this model (seed 42) | 0.157 | 0.256 | 0.243 / 0.109 / 0.118 | about 11 ms |
| YOLOv8n, mean of 3 seeds | 0.129 (range 0.108 to 0.157) | n/a | cyclist 0.118 / 0.007 / 0.000 | |
| RT-DETR-L, this model (seed 42) | 0.202 | 0.338 | 0.305 / 0.173 / 0.127 | about 35 ms |
| RT-DETR-L, mean of 3 seeds | 0.192 (range 0.149 to 0.225) | n/a | cyclist 0.127 / 0.021 / 0.169 | about 33 ms |

**Reproduction.** The recorded numbers are from CUDA with Ultralytics 8.4.198. Re-evaluating on an Apple GPU with Ultralytics
8.4.115, the YOLOv8n weights score mAP 0.1567 (recorded 0.1566) and the RT-DETR-L weights 0.193 (recorded 0.202). The RT-DETR-L
export is exact: the original training checkpoint scores the same 0.193 on the same machine, so the 0.009 difference comes
from the evaluation environment, not the export.

Seed 42 is the best of the three YOLOv8n seeds, so expect about 0.13 mAP from a retrained YOLOv8n; for RT-DETR-L
it is the middle seed, close to the mean. Do not rely on cyclist detection from either model: cyclist AP varied a
lot between seeds. Latency was measured on the RTX 4080 SUPER. Full numbers and the comparison with other models
are in the protocol.

## Intended use and limits

- Intended for research and benchmarking with this toolkit, on non-commercial terms.
- Trained on a small slice of the data (100 of 798 training segments), FRONT camera only. They miss most small
  and distant objects (detector misses account for 90 to 97% of missed objects in the tracking analysis of the
  YOLOv8n detector).
- Not validated for any safety-relevant use. They must not be used to operate or assist in operating a vehicle.

## Use

```python
import cv2, torch, yaml
from models import build_detector

cfg = yaml.safe_load(open("configs/yolov8n_hires_100.yaml"))          # or configs/rtdetr_hires_100.yaml
cfg["resume_weights"] = "yolov8n_hires_100-v1.2-s42.weights.pt"       # or rtdetr_l_hires_100-v1.2-s42.weights.pt
detector = build_detector(cfg)
model = detector.build_model(cfg, torch.device("cpu")).eval()
for det in detector.predict(model, cv2.imread("frame.jpg"), cfg):      # BGR image
    print(det["box"], det["label"], det["score"])                       # xyxy in original pixels
```

To score a model, put the file at `<run-dir>/checkpoints/latest.pt` and run `evaluate.py` as described in the
[reproducibility notes](reproducibility.md) (on Apple GPUs RT-DETR needs `PYTORCH_ENABLE_MPS_FALLBACK=1`).
Training checkpoints can be converted with `scripts/export_weights.py`.

## Cite

```bibtex
@misc{waymo_open_dataset,
  title = {Waymo Open Dataset: An autonomous driving dataset},
  website = {\url{https://www.waymo.com/open}},
  year = {2019-2025}
}
```
