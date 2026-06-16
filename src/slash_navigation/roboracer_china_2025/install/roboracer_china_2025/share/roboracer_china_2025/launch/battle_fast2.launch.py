from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('drive_topic', default_value='/drive'),
        DeclareLaunchArgument('marker_topic', default_value='/battle_fast2/arrow_marker'),
        DeclareLaunchArgument('debug_scan_topic', default_value='/battle_fast2/front_scan'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('publish_marker', default_value='true'),
        DeclareLaunchArgument('publish_debug_scan', default_value='false'),
        DeclareLaunchArgument('marker_frame', default_value=''),

        Node(
            package='roboracer_china_2025',
            executable='battle_fast2_node',
            name='battle_fast2_node',
            output='screen',
            parameters=[{
                'scan_topic': LaunchConfiguration('scan_topic'),
                'drive_topic': LaunchConfiguration('drive_topic'),
                'marker_topic': LaunchConfiguration('marker_topic'),
                'debug_scan_topic': LaunchConfiguration('debug_scan_topic'),
                'odom_topic': LaunchConfiguration('odom_topic'),
                'publish_marker': ParameterValue(LaunchConfiguration('publish_marker'), value_type=bool),
                'publish_debug_scan': ParameterValue(LaunchConfiguration('publish_debug_scan'), value_type=bool),
                'marker_frame': LaunchConfiguration('marker_frame'),
            }],
        ),
    ])
