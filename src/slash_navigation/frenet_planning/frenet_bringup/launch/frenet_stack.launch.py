import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def package_config(package_name, filename):
    return os.path.join(
        get_package_share_directory(package_name),
        "config",
        filename,
    )


def generate_launch_description():
    odom_topic = LaunchConfiguration("odom_topic")
    scan_topic = LaunchConfiguration("scan_topic")
    drive_topic = LaunchConfiguration("drive_topic")
    centerline_csv = LaunchConfiguration("centerline_csv")
    global_frame = LaunchConfiguration("global_frame")
    car_frame = LaunchConfiguration("car_frame")
    planner_start_delay_s = LaunchConfiguration("planner_start_delay_s")
    mpc_start_delay_s = LaunchConfiguration("mpc_start_delay_s")

    ego_params_file = package_config("frenet_runtime", "frenet_runtime.yaml")
    advisor_params_file = package_config("reactive_racing", "reactive_advisor.yaml")
    planner_params_file = package_config(
        "frenet_local_planner",
        "frenet_local_planner.yaml",
    )
    mpc_params_file = package_config("frenet_mpc", "frenet_mpc.yaml")

    ego_node = Node(
        package="frenet_runtime",
        executable="ego_frenet_node",
        name="ego_frenet_node",
        output="screen",
        parameters=[
            ego_params_file,
            {
                "odom_topic": odom_topic,
                "ego_state_topic": "/frenet/ego_state",
                "centerline_csv": centerline_csv,
                "global_frame": global_frame,
            },
        ],
    )

    advisor_node = Node(
        package="reactive_racing",
        executable="reactive_advisor_node",
        name="reactive_advisor_node",
        output="screen",
        parameters=[
            advisor_params_file,
            {
                "scan_topic": scan_topic,
                "advice_topic": "/frenet/reactive_advice",
            },
        ],
    )

    planner_node = Node(
        package="frenet_local_planner",
        executable="frenet_local_planner_node",
        name="frenet_local_planner_node",
        output="screen",
        parameters=[
            planner_params_file,
            {
                "ego_state_topic": "/frenet/ego_state",
                "reactive_advice_topic": "/frenet/reactive_advice",
                "planner_state_topic": "/frenet/planner_state",
                "local_trajectory_topic": "/frenet/local_trajectory",
                "local_path_topic": "/frenet/local_path",
                "centerline_csv": centerline_csv,
                "global_frame": global_frame,
            },
        ],
    )

    mpc_node = Node(
        package="frenet_mpc",
        executable="frenet_mpc_node",
        name="frenet_mpc_node",
        output="screen",
        parameters=[
            mpc_params_file,
            {
                "odom_topic": odom_topic,
                "local_trajectory_topic": "/frenet/local_trajectory",
                "drive_topic": drive_topic,
                "predicted_path_topic": "/frenet_mpc/predicted_path",
                "ref_path_topic": "/frenet_mpc/ref_path",
                "global_frame": global_frame,
                "car_frame": car_frame,
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "odom_topic",
                default_value="/pf/pose/odom",
                description="External odometry topic from particle filter.",
            ),
            DeclareLaunchArgument(
                "scan_topic",
                default_value="/scan",
                description="External LaserScan topic.",
            ),
            DeclareLaunchArgument(
                "drive_topic",
                default_value="/drive",
                description="Ackermann drive command topic published by frenet_mpc.",
            ),
            DeclareLaunchArgument(
                "centerline_csv",
                default_value="",
                description="Centerline CSV path. Empty uses each node default fallback.",
            ),
            DeclareLaunchArgument(
                "global_frame",
                default_value="map",
                description="Global frame for debug paths and Frenet projection.",
            ),
            DeclareLaunchArgument(
                "car_frame",
                default_value="base_link",
                description="Vehicle frame used by frenet_mpc debug outputs.",
            ),
            DeclareLaunchArgument(
                "planner_start_delay_s",
                default_value="1.0",
                description="Delay before starting frenet_local_planner_node.",
            ),
            DeclareLaunchArgument(
                "mpc_start_delay_s",
                default_value="2.0",
                description="Delay before starting frenet_mpc_node.",
            ),
            ego_node,
            advisor_node,
            TimerAction(
                period=planner_start_delay_s,
                actions=[planner_node],
            ),
            TimerAction(
                period=mpc_start_delay_s,
                actions=[mpc_node],
            ),
        ]
    )
