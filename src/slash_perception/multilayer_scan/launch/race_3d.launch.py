import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    全栈竞速链（W1+W2+W3），不含整车 bringup（雷达/p2l/vesc/mux 另行启动）：

        /livox/lidar --[multilayer_scan]--> /scan_3d
            --[battle_fast2 (scan_topic:=/scan_3d)]--> /drive_raw
            --[ggv_shield]--> /drive --> ackermann_mux(navigation)

    前置：W1 底座已在运行，建议
        ros2 launch roboracer_china_2025 race_reactive_bringup.launch.py run_planner:=false
    （即起雷达/地面分割/p2l/底盘/odom/mux，但不起反应式规划器，规划器由本 launch 接管）。
    本 launch 用 /scan_3d 取代单层 /scan 喂规划器；现有单层 /scan 不受影响（Nav2 等可继续用）。
    """
    ml_share = get_package_share_directory('multilayer_scan')
    roboracer_share = get_package_share_directory('roboracer_china_2025')
    shield_share = get_package_share_directory('slash_safety')

    return LaunchDescription([
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('input_topic', default_value='/livox/lidar'),
        DeclareLaunchArgument('fused_scan_topic', default_value='/scan_3d'),
        DeclareLaunchArgument('raw_drive_topic', default_value='/drive_raw'),
        DeclareLaunchArgument('out_drive_topic', default_value='/drive'),
        DeclareLaunchArgument('ml_params',
                              default_value=os.path.join(ml_share, 'config', 'multilayer_scan.yaml')),
        DeclareLaunchArgument('shield_params',
                              default_value=os.path.join(shield_share, 'config', 'ggv_shield.yaml')),

        # 1) 三维多层感知 -> /scan_3d
        Node(
            package='multilayer_scan',
            executable='multilayer_scan_node',
            name='multilayer_scan_node',
            output='screen',
            parameters=[
                LaunchConfiguration('ml_params'),
                {
                    'input_topic': LaunchConfiguration('input_topic'),
                    'fused_scan_topic': LaunchConfiguration('fused_scan_topic'),
                },
            ],
        ),

        # 2) 反应式规划器：吃 /scan_3d，发 /drive_raw
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(roboracer_share, 'launch', 'battle_fast2.launch.py')),
            launch_arguments={
                'scan_topic': LaunchConfiguration('fused_scan_topic'),
                'drive_topic': LaunchConfiguration('raw_drive_topic'),
                'odom_topic': LaunchConfiguration('odom_topic'),
            }.items(),
        ),

        # 3) g-g-v 安全护盾：/drive_raw -> /drive
        Node(
            package='slash_safety',
            executable='ggv_shield_node',
            name='ggv_shield_node',
            output='screen',
            parameters=[
                LaunchConfiguration('shield_params'),
                {
                    'drive_in_topic': LaunchConfiguration('raw_drive_topic'),
                    'drive_out_topic': LaunchConfiguration('out_drive_topic'),
                    'odom_topic': LaunchConfiguration('odom_topic'),
                },
            ],
        ),
    ])
