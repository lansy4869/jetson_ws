import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """只启动障碍运动学跟踪/测速节点（创新点2）。"""
    default_params = os.path.join(
        get_package_share_directory('obstacle_tracker'), 'config', 'obstacle_tracker.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('scan_topic', default_value='/scan_3d'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        Node(
            package='obstacle_tracker',
            executable='obstacle_tracker_node',
            name='obstacle_tracker_node',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'scan_topic': LaunchConfiguration('scan_topic'),
                    'odom_topic': LaunchConfiguration('odom_topic'),
                },
            ],
        ),
    ])
