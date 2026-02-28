# Waymo Open Dataset Toolkit

## Description


## Getting Started

To get started with Waymo Open Dataset, ensure you have gained access to the dataset using your Google account. Proceed only after you are able to view the dataset on the Google Cloud Console [here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_2_0_0).

## Install Gcloud
- Follow the instructions on this [page](https://cloud.google.com/sdk/docs/install) to install the gcloud CLI.
- Authenticate with your account via the CLI by following this [link](https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev). This ultimately should create a credentials file and stored on your development machine. These credentials will be utilized by the script to download the data.

## Download Data
- Assuming you have authenticated, creadentials are generated and accessible across applications on your development machine; run the following script:
`./scripts/download_data.sh <source-blob> <destination-folder> <-m : for parallelization>`. 
- For example, if you wish to download just `camera_image` then the command looks like this: `./scripts/download_data.sh waymo_open_dataset_v_2_0_0/training/camera_image /mnt/e/WaymoOpenDatasetV2/training/camera_image -m`
- If you wish to download the entire dataset then it is roughly `2.29TB`. You may query with `gsutil du -s -ah gs://waymo_open_dataset_v_2_0_0` if there has been any change to the dataset.

## Analyze Data

Open `notebooks/eda.ipynb` in Jupyter to explore a segment interactively.
The notebook streams data directly from GCS — no local download needed.

### Quick start

```bash
# 1. Install dependencies (see requirements.txt note on macOS)
pip install -r requirements.txt

# 2. Authenticate with GCS
gcloud auth application-default login

# 3. Launch Jupyter
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

### Programmatic usage (scripts / main.py)

```bash
# List available segments
python main.py --split training --list

# Extract camera images + 3-D LiDAR labels for one segment
python main.py --segment <context_name> --save-dir ./output

# Also extract LiDAR point cloud pickles
python main.py --segment <context_name> --save-dir ./output --all
```

## License
Licensed under [GNU AGPL v3](https://github.com/KushalBKusram/WaymoDataToolkit/blob/master/LICENSE).

 
