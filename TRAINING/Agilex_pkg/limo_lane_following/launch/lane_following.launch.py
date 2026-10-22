#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    # Base path where your scripts are located
    base_path = '/home/agilex/limo_code_StutiRuparel/bev2'

    return LaunchDescription([
        # ==========================================================
        # HARDWARE & SENSORS (Commented out as requested)
        # ==========================================================
        
        # 1. Limo Base
        # ExecuteProcess(cmd=['ros2', 'launch', 'limo_base', 'limo_base.launch.py'], output='screen'),
        
        # 2. Orbbec Astra Camera
        # ExecuteProcess(cmd=['ros2', 'launch', 'astra_camera', 'astra.launch.py'], output='screen'),
        
        # 3. YDLidar
        # ExecuteProcess(cmd=['ros2', 'launch', 'ydlidar_ros2_driver', 'ydlidar.launch.py'], output='screen'),
        
        # 4. Segmentation Node
        # ExecuteProcess(cmd=['python3', f'{base_path}/limo_segmentation_node.py'], output='screen'),

        # ==========================================================
        # LANE FOLLOWING PIPELINE
        # ==========================================================
        
        # Layer 2: Metric Ground Frame Reference
        ExecuteProcess(
            cmd=['python3', f'{base_path}/camera_metric_reference.py'],
            output='screen'
        ),
        
        # Layer 3: Lane State Filter
        ExecuteProcess(
            cmd=['python3', f'{base_path}/lane_state_filter.py'],
            output='screen'
        ),
        
        # Layer 4: Metric MPC Controller
        ExecuteProcess(
            cmd=['python3', f'{base_path}/metric_mpc_controller.py'],
            output='screen'
        ),
        
        # Layer 5: Safety Supervisor (Outputs to /cmd_vel)
        ExecuteProcess(
            cmd=['python3', f'{base_path}/safety_supervisor.py'],
            output='screen'
        ),
    ])