import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # -------------------------------------------------------------
    # 1. Base Hardware Launchers
    # -------------------------------------------------------------
    limo_base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('limo_base'),
                'launch',
                'limo_base.launch.py'
            )
        )
    )

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('orbbec_camera'),
                'launch',
                'dabai_dcw2.launch.py'
            )
        )
    )

    ydlidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ydlidar_ros2_driver'),
                'launch',
                'ydlidar_launch.py'
            )
        )
    )

    # -------------------------------------------------------------
    # 2. Autonomous Lane-Following Pipeline Nodes
    # -------------------------------------------------------------
    segmentation_node = ExecuteProcess(
        cmd=[
            'python3',
            '/home/agilex/limo_code_StutiRuparel/bev2/limo_segmentation_node.py',
            '--ros-args',
            '-p', 'checkpoint:=/home/agilex/limo_code_StutiRuparel/checkpoints/best_resnet34_unet_jaccard.pth',
            '-p', 'camera_topic:=/camera/color/image_raw',
            '-p', 'mask_topic:=/seg/mask',
            '-p', 'mask_raw_topic:=/seg/mask_raw',
            '-p', 'overlay_topic:=/seg/cam_overlay',
            '-p', 'label_topic:=/limo/camera/label_image',
            '-p', 'model_input_width:=320',
            '-p', 'model_input_height:=240',
            '-p', 'bev_mask_width:=320',
            '-p', 'bev_mask_height:=240',
            '-p', 'calibrate_mode:=True'
        ],
        output='screen'
    )

    camera_metric_reference_node = ExecuteProcess(
        cmd=[
            'python3',
            '/home/agilex/limo_code_StutiRuparel/bev2/camera_metric_reference.py'
        ],
        output='screen'
    )

    lane_state_filter_node = ExecuteProcess(
        cmd=[
            'python3',
            '/home/agilex/limo_code_StutiRuparel/bev2/lane_state_filter.py'
        ],
        output='screen'
    )

    safety_supervisor_node = ExecuteProcess(
        cmd=[
            'python3',
            '/home/agilex/limo_code_StutiRuparel/bev2/safety_supervisor.py'
        ],
        output='screen'
    )

    metric_mpc_controller_node = ExecuteProcess(
        cmd=[
            'python3',
            '/home/agilex/limo_code_StutiRuparel/bev2/metric_mpc_controller.py'
        ],
        output='screen'
    )

    # Optional: rqt_image_view for monitoring
    rqt_image_view_node = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_image_view',
        output='screen'
    )

    return LaunchDescription([
        limo_base_launch,
        camera_launch,
        ydlidar_launch,
        segmentation_node,
        camera_metric_reference_node,
        lane_state_filter_node,
        safety_supervisor_node,
        metric_mpc_controller_node,
        rqt_image_view_node
    ])