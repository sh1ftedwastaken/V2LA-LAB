# LIMO Autonomous Lane-Following System

This repository implements perception, metric reference extraction, safety supervision, and model predictive control (MPC) for autonomous lane following on the AgileX LIMO robot.

---

## Prerequisites & Network Configuration

To prevent ROS 2 node discovery and multicast communication limits on the local loopback interface, run:

```bash
sudo ip link set lo multicast on
```

---

## Option 1: Unified Launch (Single Command)

Launch all hardware drivers, perception modules, and control nodes at once using the unified launch file:

```bash
source /opt/ros/foxy/setup.bash
source ~/limo_code_StutiRuparel/install/setup.bash
source ~/limo_code_StutiRuparel/src/install/setup.bash
ros2 launch limo_line_following line_following.launch.py
```

---

## Option 2: Multi-Terminal Manual Execution

For modular testing, monitoring, and debugging, run each component in a dedicated terminal window:

### Terminal 0: Network Multicast Setup
```bash
sudo ip link set lo multicast on
```

### Terminal 1: LIMO Chassis Base Driver
```bash
source ~/limo_code_StutiRuparel/install/setup.bash
ros2 launch limo_base limo_base.launch.py
```

### Terminal 2: Orbbec DaBai RGB-D Camera Driver
```bash
source ~/limo_code_StutiRuparel/install/setup.bash
ros2 launch orbbec_camera dabai_dcw2.launch.py
```

### Terminal 3: YDLIDAR Laser Scanner Driver
```bash
source ~/limo_code_StutiRuparel/install/setup.bash
source ~/limo_code_StutiRuparel/src/install/setup.bash
ros2 launch ydlidar_ros2_driver ydlidar_launch.py
```

### Terminal 4: Neural Segmentation Node
Runs inference using the ResNet-34 U-Net model and outputs raw BEV masks [3]:
```bash
source /opt/ros/foxy/setup.bash
python3 /home/agilex/limo_code_StutiRuparel/limo_line_following/limo_segmentation_node.py \
--ros-args \
-p checkpoint:=/home/agilex/limo_code_StutiRuparel/checkpoints/best_resnet34_unet_jaccard.pth \
-p camera_topic:=/camera/color/image_raw \
-p mask_topic:=/seg/mask \
-p mask_raw_topic:=/seg/mask_raw \
-p overlay_topic:=/seg/cam_overlay \
-p label_topic:=/limo/camera/label_image \
-p model_input_width:=320 \
-p model_input_height:=240 \
-p bev_mask_width:=320 \
-p bev_mask_height:=240 \
-p calibrate_mode:=True
```

### Terminal 5: Visualization
```bash
source /opt/ros/foxy/setup.bash
ros2 run rqt_image_view rqt_image_view
```
*(Select `/seg/cam_overlay` [3] or `/lane/camera_reference_debug` [1] to view tracking performance).*

### Terminal 6: Camera Metric Reference Node
Projects segmented mask boundaries into metric ground coordinates and extracts corridor centerlines [1]:
```bash
source /opt/ros/foxy/setup.bash
python3 /home/agilex/limo_code_StutiRuparel/limo_line_following/camera_metric_reference.py
```

### Terminal 7: Lane State Diagnostic Filter
Calculates adaptive lookahead lateral displacement, heading, and curvature [2]:
```bash
source /opt/ros/foxy/setup.bash
python3 /home/agilex/limo_code_StutiRuparel/limo_line_following/lane_state_filter.py
```

### Terminal 8: Safety Supervisor
Monitors laser scan obstacles and gates commands to the chassis base [5]:
```bash
source /opt/ros/foxy/setup.bash
python3 /home/agilex/limo_code_StutiRuparel/limo_line_following/safety_supervisor.py
```

### Terminal 9: Metric MPC Controller
Computes dynamically constrained velocity and steering commands along the reference trajectory [4]:
```bash
source /opt/ros/foxy/setup.bash
python3 /home/agilex/limo_code_StutiRuparel/limo_line_following/metric_mpc_controller.py
```

---

## Architecture & Data Flow

- **Segmentation [3]:** Receives `/camera/color/image_raw` and publishes bird's-eye view masks to `/seg/mask_raw` [3].
- **Metric Reference [1]:** Computes metric coordinates and publishes paths to `/lane/metric_reference` and `/lane/metric_reference_debug` [1].
- **Lane State Diagnostic [2]:** Consumes `/lane/metric_reference` and outputs polynomial road curvature and heading estimates to `/lane/lane_state_debug` [2].
- **MPC Tracking [4]:** Solves constrained horizon trajectories to generate smooth linear and angular velocity commands on `/cmd_vel_request` [4].
- **Safety Supervision [5]:** Subscribes to `/cmd_vel_request` and `/scan`, stopping the robot if obstacles are closer than 0.30 m or if reference updates stall, forwarding safe commands to `/cmd_vel` [5].
```

Would you like me to explain or break down the code?