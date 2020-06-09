# WaymoDataToolkit v1.0

## Description
A set of functions to extract raw data from tfrecord segment files stored in Waymo Google Cloud Bucket. 

## Features of v1.0
- Extract images per frame per segment with corresponding labels
- Extract images per camera with corresponding labels
- Extracted images are stored as jpeg
- Extracted labels are in the format: object-class x y width height

## Future updates
- Visualize images as they are being extracted
- Extract LiDAR data
- Visualize LiDAR data

## Requirements
Ubuntu 18.04, Python 3, Tensorflow, Numpy, Waymo Open Dataset

## Usage
Repo consists [src/demo.py](https://github.com/KushalBKusram/WaymoDataToolkit/src/demo.py) which has code to get you started once you have setup your environment with required libraries, gained access to Waymo data and bucket.

## License
Licensed under [GNU AGPL v3](https://github.com/KushalBKusram/HoloNav/blob/master/LICENSE).

 
