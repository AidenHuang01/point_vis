# Point_vis
## Author: Yucheng Huang

## Introduction of this project:
This project is to help improve 3D autolabeling of lidar data using sensor 
fusion projection that combines 3D sapce and 2D space. You can input your 
3D bounding boxes and use the 2D detection boxes result from camera to 
determine outlier 3D bounding boxes in order to achieve higher accuracy in 
3D object detection. The refined 3D boxes from lidar can later be used as 
ground truth training and validation data for radar perception model.

## Usage (demo)
```bash
git clone git@github.com:AidenHuang01/point_vis.git
cd point_vis/projection
python ./single_frame_fusion.py
```

## Usage (your data)
* put your image into ./point_vis/projection/data/image (optional)
* put your point cloud data (.npy) into ./point_vis/projection/data/pcd (optional)
* put your 2D bounding boxes vertices into ./point_vis/projection/data/box2d
* put your 3D bounding boxes center into ./point_vis/projection/data/box3d
* edit your calib information from lidar to camera in single_frame_fusion.py::lidar2CameraOurs()  
  
```bash
cd point_vis/projection
python ./single_frame_fusion.py
```
