# Waymo Open Dataset Toolkit

## Description
A set of functions to extract and visualize Waymo Open Dataset. 

## Features
- Extract images per frame per segment with corresponding labels
- Extract images per camera with corresponding labels
- Extracted images are stored as png
- Extracted labels are in the format: object-class x y width height
- Extract LiDAR data as point clouds with camera projections
- Visualize LiDAR data as point cloud

## Screenshots

### Camera Data
![Camera Data](images\camera.png)

### Point Cloud Data
![Point Cloud Data](images\lidar.gif)

## Requirements
Linux, Python, Waymo Open Dataset, OpenCV, Open3D

## Usage
Repo consists [src/demo.py](src/demo.py) which has code to get you started once you have setup your environment with required libraries, gained access to Waymo data and bucket.

## License
Licensed under [GNU AGPL v3](https://github.com/KushalBKusram/WaymoDataToolkit/blob/master/LICENSE).

 
