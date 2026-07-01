import json
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


MPC_NOMINAL_TOPIC = "/mpc/drive_nominal"
BATTLE_REACTIVE_TOPIC = "/battle_fast2/drive_reactive"
FRONT_CLEARANCE_TOPIC = "/battle_fast2/front_clearance_m"
RISK_MIN_MARGIN_TOPIC = "/battle_fast2/risk_min_margin_m"
REACTIVE_SPEED_LIMIT_TOPIC = "/battle_fast2/reactive_speed_limit_mps"


def _write_mpc_overrides(context):
    waypoint_csv = LaunchConfiguration("waypoint_csv").perform(context)
    trajectory_mode = LaunchConfiguration("trajectory_mode").perform(context)
    odom_topic = LaunchConfiguration("odom_topic").perform(context)
    max_speed = LaunchConfiguration("max_speed").perform(context)
    max_steer = LaunchConfiguration("max_steer").perform(context)

    overrides = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="global_mpc_overrides_",
        suffix=".yaml",
        delete=False,
    )
    with overrides:
        overrides.write("mpc_control:\n")
        overrides.write("  ros__parameters:\n")
        overrides.write(f"    drive_topic: {json.dumps(MPC_NOMINAL_TOPIC)}\n")
        overrides.write(f"    odom_topic: {json.dumps(odom_topic)}\n")
        overrides.write(f"    max_speed: {float(max_speed)}\n")
        overrides.write(f"    max_steer: {float(max_steer)}\n")
        if waypoint_csv:
            overrides.write(f"    waypoint_csv: {json.dumps(waypoint_csv)}\n")
        if trajectory_mode:
            overrides.write(f"    trajectory_mode: {json.dumps(trajectory_mode)}\n")
    return overrides.name


def _launch_setup(context, *args, **kwargs):
    mpc_params_file = LaunchConfiguration("mpc_params_file").perform(context)
    mpc_overrides = _write_mpc_overrides(context)

    scan_topic = LaunchConfiguration("scan_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    drive_topic = LaunchConfiguration("drive_topic")
    global_frame = LaunchConfiguration("global_frame")
    max_speed = LaunchConfiguration("max_speed")
    max_steer = LaunchConfiguration("max_steer")
    command_timeout_s = LaunchConfiguration("command_timeout_s")
    odom_timeout_s = LaunchConfiguration("odom_timeout_s")
    enable_safety_shield = LaunchConfiguration("enable_safety_shield")
    shield_diag_timeout_s = LaunchConfiguration("shield_diag_timeout_s")
    shield_yellow_margin_m = LaunchConfiguration("shield_yellow_margin_m")
    shield_orange_margin_m = LaunchConfiguration("shield_orange_margin_m")
    shield_red_margin_m = LaunchConfiguration("shield_red_margin_m")
    shield_black_clearance_m = LaunchConfiguration("shield_black_clearance_m")
    shield_orange_blend = LaunchConfiguration("shield_orange_blend")

    return [
        Node(
            package="mpc_control",
            executable="mpc_node",
            name="mpc_control",
            output="screen",
            parameters=[mpc_params_file, mpc_overrides],
        ),
        Node(
            package="roboracer_china_2025",
            executable="battle_fast2_node",
            name="battle_fast2_node",
            output="screen",
            parameters=[
                {
                    "scan_topic": scan_topic,
                    "drive_topic": BATTLE_REACTIVE_TOPIC,
                    "front_clearance_topic": FRONT_CLEARANCE_TOPIC,
                    "risk_min_margin_topic": RISK_MIN_MARGIN_TOPIC,
                    "reactive_speed_limit_topic": REACTIVE_SPEED_LIMIT_TOPIC,
                    "reachability_max_speed": ParameterValue(max_speed, value_type=float),
                    "reachability_max_steer": ParameterValue(max_steer, value_type=float),
                }
            ],
        ),
        Node(
            package="roboracer_china_2025",
            executable="drive_arbiter",
            name="drive_arbiter",
            output="screen",
            parameters=[
                {
                    "mpc_drive_topic": MPC_NOMINAL_TOPIC,
                    "reactive_drive_topic": BATTLE_REACTIVE_TOPIC,
                    "odom_topic": odom_topic,
                    "drive_topic": drive_topic,
                    "global_frame": global_frame,
                    "front_clearance_topic": FRONT_CLEARANCE_TOPIC,
                    "risk_min_margin_topic": RISK_MIN_MARGIN_TOPIC,
                    "reactive_speed_limit_topic": REACTIVE_SPEED_LIMIT_TOPIC,
                    "max_speed": ParameterValue(max_speed, value_type=float),
                    "max_steer": ParameterValue(max_steer, value_type=float),
                    "command_timeout_s": ParameterValue(command_timeout_s, value_type=float),
                    "odom_timeout_s": ParameterValue(odom_timeout_s, value_type=float),
                    "enable_safety_shield": ParameterValue(
                        enable_safety_shield,
                        value_type=bool,
                    ),
                    "shield_diag_timeout_s": ParameterValue(
                        shield_diag_timeout_s,
                        value_type=float,
                    ),
                    "shield_yellow_margin_m": ParameterValue(
                        shield_yellow_margin_m,
                        value_type=float,
                    ),
                    "shield_orange_margin_m": ParameterValue(
                        shield_orange_margin_m,
                        value_type=float,
                    ),
                    "shield_red_margin_m": ParameterValue(
                        shield_red_margin_m,
                        value_type=float,
                    ),
                    "shield_black_clearance_m": ParameterValue(
                        shield_black_clearance_m,
                        value_type=float,
                    ),
                    "shield_orange_blend": ParameterValue(
                        shield_orange_blend,
                        value_type=float,
                    ),
                }
            ],
        ),
    ]


def generate_launch_description():
    mpc_dir = get_package_share_directory("mpc_control")

    return LaunchDescription(
        [
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            DeclareLaunchArgument("odom_topic", default_value="/pf/pose/odom"),
            DeclareLaunchArgument("drive_topic", default_value="/drive"),
            DeclareLaunchArgument("global_frame", default_value="map"),
            DeclareLaunchArgument("waypoint_csv", default_value=""),
            DeclareLaunchArgument(
                "mpc_params_file",
                default_value=os.path.join(mpc_dir, "config", "mpc_control.yaml"),
            ),
            DeclareLaunchArgument("trajectory_mode", default_value=""),
            DeclareLaunchArgument("max_speed", default_value="2.0"),
            DeclareLaunchArgument("max_steer", default_value="0.412"),
            DeclareLaunchArgument("command_timeout_s", default_value="0.2"),
            DeclareLaunchArgument("odom_timeout_s", default_value="0.3"),
            DeclareLaunchArgument("enable_safety_shield", default_value="true"),
            DeclareLaunchArgument("shield_diag_timeout_s", default_value="0.2"),
            DeclareLaunchArgument("shield_yellow_margin_m", default_value="0.55"),
            DeclareLaunchArgument("shield_orange_margin_m", default_value="0.35"),
            DeclareLaunchArgument("shield_red_margin_m", default_value="0.18"),
            DeclareLaunchArgument("shield_black_clearance_m", default_value="0.30"),
            DeclareLaunchArgument("shield_orange_blend", default_value="0.55"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
