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
        DeclareLaunchArgument('use_reachability_core', default_value='true'),
        DeclareLaunchArgument('reachability_fallback_to_original', default_value='true'),
        DeclareLaunchArgument('reachability_max_speed', default_value='2.5'),
        DeclareLaunchArgument('reachability_vehicle_width', default_value='0.32'),
        DeclareLaunchArgument('reachability_base_margin', default_value='0.16'),
        DeclareLaunchArgument('reachability_system_delay', default_value='0.16'),
        DeclareLaunchArgument('reachability_risk_speed_gain', default_value='0.55'),
        DeclareLaunchArgument('reachability_risk_weight', default_value='3.0'),
        DeclareLaunchArgument('reachability_confidence_weight', default_value='1.0'),
        DeclareLaunchArgument('reachability_dynamic_risk_weight', default_value='0.65'),
        DeclareLaunchArgument('reachability_dynamic_closing_speed_threshold', default_value='0.5'),
        DeclareLaunchArgument('reachability_dynamic_risk_range', default_value='8.0'),
        DeclareLaunchArgument('reachability_dynamic_max_delta_time', default_value='0.5'),

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
                'use_reachability_core': ParameterValue(
                    LaunchConfiguration('use_reachability_core'), value_type=bool),
                'reachability_fallback_to_original': ParameterValue(
                    LaunchConfiguration('reachability_fallback_to_original'), value_type=bool),
                'reachability_max_speed': ParameterValue(
                    LaunchConfiguration('reachability_max_speed'), value_type=float),
                'reachability_vehicle_width': ParameterValue(
                    LaunchConfiguration('reachability_vehicle_width'), value_type=float),
                'reachability_base_margin': ParameterValue(
                    LaunchConfiguration('reachability_base_margin'), value_type=float),
                'reachability_system_delay': ParameterValue(
                    LaunchConfiguration('reachability_system_delay'), value_type=float),
                'reachability_risk_speed_gain': ParameterValue(
                    LaunchConfiguration('reachability_risk_speed_gain'), value_type=float),
                'reachability_risk_weight': ParameterValue(
                    LaunchConfiguration('reachability_risk_weight'), value_type=float),
                'reachability_confidence_weight': ParameterValue(
                    LaunchConfiguration('reachability_confidence_weight'), value_type=float),
                'reachability_dynamic_risk_weight': ParameterValue(
                    LaunchConfiguration('reachability_dynamic_risk_weight'), value_type=float),
                'reachability_dynamic_closing_speed_threshold': ParameterValue(
                    LaunchConfiguration('reachability_dynamic_closing_speed_threshold'), value_type=float),
                'reachability_dynamic_risk_range': ParameterValue(
                    LaunchConfiguration('reachability_dynamic_risk_range'), value_type=float),
                'reachability_dynamic_max_delta_time': ParameterValue(
                    LaunchConfiguration('reachability_dynamic_max_delta_time'), value_type=float),
            }],
        ),
    ])
