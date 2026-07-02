import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share_dir = get_package_share_directory("frenet_runtime")
    default_params_file = os.path.join(
        package_share_dir,
        "config",
        "frenet_runtime.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params_file,
            ),
            Node(
                package="frenet_runtime",
                executable="ego_frenet_node",
                name="ego_frenet_node",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
        ]
    )
