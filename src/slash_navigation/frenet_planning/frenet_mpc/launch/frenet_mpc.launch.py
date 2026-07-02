import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    odom_topic = LaunchConfiguration("odom_topic")
    local_trajectory_topic = LaunchConfiguration("local_trajectory_topic")
    drive_topic = LaunchConfiguration("drive_topic")

    default_params_file = os.path.join(
        get_package_share_directory("frenet_mpc"),
        "config",
        "frenet_mpc.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params_file,
                description="Path to the frenet_mpc parameter file.",
            ),
            DeclareLaunchArgument(
                "odom_topic",
                default_value="/pf/pose/odom",
                description="Odometry topic override.",
            ),
            DeclareLaunchArgument(
                "local_trajectory_topic",
                default_value="/frenet/local_trajectory",
                description="Local trajectory topic override.",
            ),
            DeclareLaunchArgument(
                "drive_topic",
                default_value="/drive",
                description="Ackermann drive command topic override.",
            ),
            Node(
                package="frenet_mpc",
                executable="frenet_mpc_node",
                name="frenet_mpc_node",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "odom_topic": odom_topic,
                        "local_trajectory_topic": local_trajectory_topic,
                        "drive_topic": drive_topic,
                    },
                ],
            ),
        ]
    )
