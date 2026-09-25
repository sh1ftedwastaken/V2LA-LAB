# Autonomous Driving Pipeline

This repository contains a ROS 2 based autonomous driving pipeline for vision-based lane following and obstacle avoidance. The system uses a deep learning segmentation model to identify roads, lanes, and vehicles, warps the camera view into a Bird's Eye View (BEV), and calculates real-time driving commands.

## System Architecture

The pipeline consists of the following core modules:
*   **Data Conversion (`convert_masks.py`):** Converts color `.jpg` masks into single-channel `.png` files (mapping classes 0-4), which is required before training. It automatically merges red and blue vehicles into a single vehicle class.
*   **Model Training (`train2.py` & `loss_functions.py`):** The main training script used to train the neural network (e.g., ResNet34-UNet) on the processed masks.
*   **Segmentation Node (`limo_segmentation_node.py`):** A ROS 2 node that runs the trained neural network inference on the robot's raw camera feed. It publishes the raw mono8 mask (`/seg/mask_raw`) and a colorized version for debugging (`/seg/mask`).
*   **BEV Node (`seg_bev_node.py`):** Subscribes to the raw segmentation mask and warps it into a flattened Bird's Eye View (`/seg/bev_mask`). It also publishes visual overlays (`/seg/bev_overlay`) to debug the Region of Interest (ROI) and lane centers.
*   **Autonomous Driving (`autonomous_driving.py` & `lane_analyzer.py`):** The primary controller that subscribes to the BEV masks to extract lane coordinates and generate velocity commands (`/cmd_vel`).

---

## Execution Guide

### 0. Environment Setup and Package Installation
```
# Create the main folder
mkdir -p /home/agilex/limo_code_StutiRuparel/

# (Manually paste/copy your 'bev2' folder into this directory)
# Navigate into the project folder
cd /home/agilex/limo_code_StutiRuparel/bev2
```

> **NOTE:** Ensure to source ROS 2 workspace in **every** terminal before running the commands:
>```bash
>source ~/limo_code_StutiRuparel/install/setup.bash
>source ~/opt/ros/foxy/setup.bash
>```


### 1. Limo initialization
In Terminals 1, 2, and 3, start the robot's base chassis, depth camera, and LiDAR using the combined run script:
```bash
# Terminal 1:
source ~/limo_code_StutiRuparel/install/setup.bash
source ~/opt/ros/foxy/setup.bash
ros2 launch limo_base limo_base.launch.py

#Terminal 2: 
source ~/limo_code_StutiRuparel/install/setup.bash
source ~/opt/ros/foxy/setup.bash
ros2 launch orbbec_camera dabai_dcw2.launch.py

# Terminal 3: 
source ~/opt/ros/foxy/setup.bash
source ~/limo_code_StutiRuparel/src/install/setup.bash
ros2 launch ydlidar_ros2_driver ydlidar_launch.py
```

### 2. Segmentation Node
In Terminal 4, launch the segmentation node. Point the `checkpoint` parameter to the best trained `.pth` model weights:

```bash
source ~/limo_code_StutiRuparel/install/setup.bash
source ~/opt/ros/foxy/setup.bash
python3 /home/agilex/limo_code_StutiRuparel/bev2/limo_segmentation_node.py \
--ros-args \
-p checkpoint:=/home/agilex/limo_code_StutiRuparel/checkpoints/best_resnet34_unet_jaccard.pth \
-p camera_topic:=/camera/color/image_raw \
-p mask_topic:=/seg/mask \
-p mask_raw_topic:=/seg/mask_raw \
-p overlay_topic:=/seg/cam_overlay \
-p label_topic:=/limo/camera/label_image \
-p model_input_width:=320 \
-p model_input_height:=240 \
-p bev_mask_width:=160 \
-p bev_mask_height:=120
```

### 3. Bird's Eye View (BEV) Transform Node
In Terminal 5, run the BEV projection node. You can leave `calibrate_mode:=True` on if you need to debug the source points:
```bash
source ~/limo_code_StutiRuparel/install/setup.bash
source ~/opt/ros/foxy/setup.bash
python3 /home/agilex/limo_code_StutiRuparel/bev2/seg_bev_node.py \
--ros-args \
-p mask_topic:=/seg/mask_raw \
-p camera_params:=/home/agilex/limo_code_StutiRuparel/bev2/camera_params.txt \
-p bev_size:=160 \
-p mask_width:=160 \
-p mask_height:=120 \
-p calibrate_mode:=True         # (OPTIONAL) Keep for debugging
``` 

### 4. Visualization & Debugging
To monitor what the robot sees and how it calculates lanes, open two visualization tools.
**Terminal 6 (RQT Viewer):**
```bash
source ~/limo_code_StutiRuparel/install/setup.bash
source ~/opt/ros/foxy/setup.bash
ros2 run rqt_image_view rqt_image_view
```
**Terminal 7 (Custom Viewer - All combined):**
```bash
source ~/limo_code_StutiRuparel/install/setup.bash
source ~/opt/ros/foxy/setup.bash
python3 /home/agilex/limo_code_StutiRuparel/bev2/debug_visualizer.py
```

### 5. Autonomous Driving Controller
> **IMPORTANT:** Start the script when the car camera is looking at a **straight road**. Do **NOT** start the car **inside a curve** or at the **beginning of a curve**.


In Terminal 8, activate the main driving script to start moving. You can adjust the top speed using the `max_speed` parameter:
```bash
source ~/limo_code_StutiRuparel/install/setup.bash
source ~/opt/ros/foxy/setup.bash
python3 /home/agilex/limo_code_StutiRuparel/bev2/autonomous_driving.py \
--ros-args \
-p max_speed:=0.25
```