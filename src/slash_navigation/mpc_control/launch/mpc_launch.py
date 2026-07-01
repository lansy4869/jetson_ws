import json
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    params_file = LaunchConfiguration("params_file").perform(context)
    waypoint_csv = LaunchConfiguration("waypoint_csv").perform(context)
    trajectory_mode = LaunchConfiguration("trajectory_mode").perform(context)
    odom_topic = LaunchConfiguration("odom_topic").perform(context)
    drive_topic = LaunchConfiguration("drive_topic").perform(context)

    parameters = [params_file]
    if any([waypoint_csv, trajectory_mode, odom_topic, drive_topic]):
        overrides = tempfile.NamedTemporaryFile(
            mode="w", prefix="mpc_params_", suffix=".yaml", delete=False
        )
        with overrides:
            overrides.write("mpc_control:\n")
            overrides.write("  ros__parameters:\n")
            if waypoint_csv:
                overrides.write(f"    waypoint_csv: {json.dumps(waypoint_csv)}\n")
            if trajectory_mode:
                overrides.write(f"    trajectory_mode: {json.dumps(trajectory_mode)}\n")
            if odom_topic:
                overrides.write(f"    odom_topic: {json.dumps(odom_topic)}\n")
            if drive_topic:
                overrides.write(f"    drive_topic: {json.dumps(drive_topic)}\n")
        parameters.append(overrides.name)

    return [
        Node(
            package="mpc_control",
            executable="mpc_node",
            name="mpc_control",
            output="screen",
            parameters=parameters,
        )
    ]


def generate_launch_description():
    mpc_dir = get_package_share_directory("mpc_control")
    csv_data_dir = get_package_share_directory("csv_data")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(mpc_dir, "config", "mpc_control.yaml"),
            ),
            DeclareLaunchArgument(
                "waypoint_csv",
                default_value=os.path.join(csv_data_dir, "8flab.csv"),
                description="Waypoint CSV override.",
            ),
            DeclareLaunchArgument(
                "trajectory_mode",
                default_value="",
                description="Optional trajectory mode override; empty uses params_file.",
            ),
            DeclareLaunchArgument(
                "odom_topic",
                default_value="",
                description="Optional odom topic override; empty uses params_file.",
            ),
            DeclareLaunchArgument(
                "drive_topic",
                default_value="",
                description="Optional drive topic override; empty uses params_file.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
