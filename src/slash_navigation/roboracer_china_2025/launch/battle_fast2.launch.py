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
        DeclareLaunchArgument('front_clearance_topic', default_value='/battle_fast2/front_clearance_m'),
        DeclareLaunchArgument('risk_min_margin_topic', default_value='/battle_fast2/risk_min_margin_m'),
        DeclareLaunchArgument('reactive_speed_limit_topic', default_value='/battle_fast2/reactive_speed_limit_mps'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('publish_marker', default_value='true'),
        DeclareLaunchArgument('publish_debug_scan', default_value='false'),
        DeclareLaunchArgument('marker_frame', default_value=''),
        DeclareLaunchArgument('use_reachability_core', default_value='true'),
        DeclareLaunchArgument('reachability_fallback_to_original', default_value='false'),
        DeclareLaunchArgument('reachability_max_speed', default_value='2.5'),
        DeclareLaunchArgument('reachability_vehicle_width', default_value='0.29'),
        DeclareLaunchArgument('reachability_front_overhang', default_value='0.20'),
        DeclareLaunchArgument('reachability_rear_overhang', default_value='0.35'),
        DeclareLaunchArgument('reachability_base_margin', default_value='0.16'),
        DeclareLaunchArgument('reachability_system_delay', default_value='0.16'),

        # === AC-ARPS 自适应保形 Ackermann 残差预测安全层 ===
        # 部署顺序：阶段0 shadow_mode=true → 阶段1 shadow=false/speed_only=true
        # → 阶段2 speed_only=false（残差先 0.07 再 0.105）→ 阶段3 adaptive_margin=true
        DeclareLaunchArgument('shield_enabled', default_value='true'),
        DeclareLaunchArgument('shield_shadow_mode', default_value='true'),
        DeclareLaunchArgument('shield_speed_only', default_value='true'),
        DeclareLaunchArgument('shield_adaptive_margin_enabled', default_value='false'),
        DeclareLaunchArgument('shield_max_steering_residual', default_value='0.105'),
        DeclareLaunchArgument('shield_fixed_margin', default_value='0.08'),
        DeclareLaunchArgument('shield_wheelbase', default_value='0.33'),
        DeclareLaunchArgument('shield_vehicle_length', default_value='0.50'),
        DeclareLaunchArgument('shield_vehicle_width', default_value='0.30'),
        DeclareLaunchArgument('shield_laser_x_offset', default_value='0.12'),
        DeclareLaunchArgument('shield_tau_delay', default_value='0.16'),
        DeclareLaunchArgument('shield_a_brake', default_value='2.5'),
        DeclareLaunchArgument('shield_max_speed', default_value='2.7'),
        DeclareLaunchArgument('shield_ttc_emergency', default_value='0.35'),
        DeclareLaunchArgument('shield_ttc_slow', default_value='1.20'),

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
                'front_clearance_topic': LaunchConfiguration('front_clearance_topic'),
                'risk_min_margin_topic': LaunchConfiguration('risk_min_margin_topic'),
                'reactive_speed_limit_topic': LaunchConfiguration('reactive_speed_limit_topic'),
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
                'reachability_front_overhang': ParameterValue(
                    LaunchConfiguration('reachability_front_overhang'), value_type=float),
                'reachability_rear_overhang': ParameterValue(
                    LaunchConfiguration('reachability_rear_overhang'), value_type=float),
                'reachability_base_margin': ParameterValue(
                    LaunchConfiguration('reachability_base_margin'), value_type=float),
                'reachability_system_delay': ParameterValue(
                    LaunchConfiguration('reachability_system_delay'), value_type=float),
                'shield_enabled': ParameterValue(
                    LaunchConfiguration('shield_enabled'), value_type=bool),
                'shield_shadow_mode': ParameterValue(
                    LaunchConfiguration('shield_shadow_mode'), value_type=bool),
                'shield_speed_only': ParameterValue(
                    LaunchConfiguration('shield_speed_only'), value_type=bool),
                'shield_adaptive_margin_enabled': ParameterValue(
                    LaunchConfiguration('shield_adaptive_margin_enabled'), value_type=bool),
                'shield_max_steering_residual': ParameterValue(
                    LaunchConfiguration('shield_max_steering_residual'), value_type=float),
                'shield_fixed_margin': ParameterValue(
                    LaunchConfiguration('shield_fixed_margin'), value_type=float),
                'shield_wheelbase': ParameterValue(
                    LaunchConfiguration('shield_wheelbase'), value_type=float),
                'shield_vehicle_length': ParameterValue(
                    LaunchConfiguration('shield_vehicle_length'), value_type=float),
                'shield_vehicle_width': ParameterValue(
                    LaunchConfiguration('shield_vehicle_width'), value_type=float),
                'shield_laser_x_offset': ParameterValue(
                    LaunchConfiguration('shield_laser_x_offset'), value_type=float),
                'shield_tau_delay': ParameterValue(
                    LaunchConfiguration('shield_tau_delay'), value_type=float),
                'shield_a_brake': ParameterValue(
                    LaunchConfiguration('shield_a_brake'), value_type=float),
                'shield_max_speed': ParameterValue(
                    LaunchConfiguration('shield_max_speed'), value_type=float),
                'shield_ttc_emergency': ParameterValue(
                    LaunchConfiguration('shield_ttc_emergency'), value_type=float),
                'shield_ttc_slow': ParameterValue(
                    LaunchConfiguration('shield_ttc_slow'), value_type=float),
            }],
        ),
    ])
