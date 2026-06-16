import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """只启动 g-g-v 安全护盾（上游 planner / mux 另行启动）。"""
    default_params = os.path.join(
        get_package_share_directory('slash_safety'), 'config', 'ggv_shield.yaml'
    )
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('drive_in_topic', default_value='/drive_raw'),
        DeclareLaunchArgument('drive_out_topic', default_value='/drive'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        Node(
            package='slash_safety',
            executable='ggv_shield_node',
            name='ggv_shield_node',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'drive_in_topic': LaunchConfiguration('drive_in_topic'),
                    'drive_out_topic': LaunchConfiguration('drive_out_topic'),
                    'odom_topic': LaunchConfiguration('odom_topic'),
                },
            ],
        ),
    ])
