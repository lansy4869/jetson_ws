import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """只启动三维多层感知节点（输出 /scan_3d + 各层调试 scan）。"""
    default_params = os.path.join(
        get_package_share_directory('multilayer_scan'), 'config', 'multilayer_scan.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('input_topic', default_value='/livox/lidar'),
        DeclareLaunchArgument('fused_scan_topic', default_value='/scan_3d'),
        Node(
            package='multilayer_scan',
            executable='multilayer_scan_node',
            name='multilayer_scan_node',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'input_topic': LaunchConfiguration('input_topic'),
                    'fused_scan_topic': LaunchConfiguration('fused_scan_topic'),
                },
            ],
        ),
    ])
