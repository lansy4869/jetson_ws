from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # 注意：参数列表已与 battle_fast2_node.py 实际 declare 的参数严格对齐。
    # odom_topic 仍保留为 launch 参数（供 race_with_shield / race_3d 统一透传），
    # 但该反应式节点本身不订阅里程计，故不下发到节点（避免设置未声明参数）。
    return LaunchDescription([
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('drive_topic', default_value='/drive'),
        DeclareLaunchArgument('marker_topic', default_value='/battle_fast2/arrow_marker'),
        DeclareLaunchArgument('debug_scan_topic', default_value='/battle_fast2/front_scan'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),       # 兼容上层透传，节点不使用
        DeclareLaunchArgument('speed_scale', default_value='0.6'),        # W1 保守起步
        DeclareLaunchArgument('max_speed', default_value='2.0'),          # m/s 速度上限
        DeclareLaunchArgument('max_steer', default_value='0.34'),         # rad 转向限幅
        DeclareLaunchArgument('verbose', default_value='false'),          # 调试日志开关
        DeclareLaunchArgument('use_external_dynamic', default_value='true'),  # W4：用运动学动态判据
        DeclareLaunchArgument('dynamic_flag_topic', default_value='/perception/dynamic_obstacle'),

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
                'speed_scale': ParameterValue(LaunchConfiguration('speed_scale'), value_type=float),
                'max_speed': ParameterValue(LaunchConfiguration('max_speed'), value_type=float),
                'max_steer': ParameterValue(LaunchConfiguration('max_steer'), value_type=float),
                'verbose': ParameterValue(LaunchConfiguration('verbose'), value_type=bool),
                'use_external_dynamic': ParameterValue(LaunchConfiguration('use_external_dynamic'), value_type=bool),
                'dynamic_flag_topic': LaunchConfiguration('dynamic_flag_topic'),
            }],
        ),
    ])
