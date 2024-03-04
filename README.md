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

## License
Licensed under [GNU AGPL v3](https://github.com/KushalBKusram/WaymoDataToolkit/blob/master/LICENSE).

 
