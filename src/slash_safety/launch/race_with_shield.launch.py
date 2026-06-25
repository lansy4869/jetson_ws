from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """
    反应式规划器 + g-g-v 安全护盾 串联：

        battle_fast2 --(/drive_raw)--> ggv_shield --(/drive)--> ackermann_mux(navigation)

    注意：本 launch 不启动雷达 / p2l / mux / vesc（那是 W1 的整车 bringup 负责）。
    它假设 /scan 与 /odom 已经在跑，mux 的 navigation 口订阅 /drive。
    """
    shield_share = get_package_share_directory('slash_safety')
    roboracer_share = get_package_share_directory('roboracer_china_2025')

    return LaunchDescription([
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('raw_drive_topic', default_value='/drive_raw'),
        DeclareLaunchArgument('out_drive_topic', default_value='/drive'),
        # 默认值与 ggv_shield.yaml / battle_fast2.launch.py 保守档对齐：
        # speed_scale=0.6, max_speed/v_max=2.0, max_steer/steering_limit=0.34,
        # ay_safety_factor=0.6, use_odom_speed_as_ref=true。
        # 需要赛道提速时用显式 launch 参数覆盖，避免“日常安全壳”默认跑到实测边界外。
        DeclareLaunchArgument('speed_scale', default_value='0.6'),
        DeclareLaunchArgument('max_speed', default_value='2.0'),
        DeclareLaunchArgument('max_steer', default_value='0.34'),
        DeclareLaunchArgument('shield_wheelbase', default_value='0.25'),
        DeclareLaunchArgument('shield_ay_max', default_value='9.81'),
        DeclareLaunchArgument('shield_v_max', default_value='2.0'),
        DeclareLaunchArgument('shield_v_min', default_value='0.0'),
        DeclareLaunchArgument('shield_steering_limit', default_value='0.34'),
        DeclareLaunchArgument('shield_ay_safety_factor', default_value='0.6'),
        DeclareLaunchArgument('shield_ax_accel_max', default_value='6.35'),
        DeclareLaunchArgument('shield_ax_brake_max', default_value='6.66'),
        DeclareLaunchArgument('shield_control_rate_hz', default_value='50.0'),
        DeclareLaunchArgument('shield_input_timeout_s', default_value='0.3'),
        DeclareLaunchArgument('shield_odom_timeout_s', default_value='0.5'),
        DeclareLaunchArgument('use_odom_speed_as_ref', default_value='true'),

        # 1) 反应式规划器：把指令发到 /drive_raw（而非直接 /drive）
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(roboracer_share, 'launch', 'battle_fast2.launch.py')
            ),
            launch_arguments={
                'scan_topic': LaunchConfiguration('scan_topic'),
                'drive_topic': LaunchConfiguration('raw_drive_topic'),
                'odom_topic': LaunchConfiguration('odom_topic'),
                'speed_scale': LaunchConfiguration('speed_scale'),
                'max_speed': LaunchConfiguration('max_speed'),
                'max_steer': LaunchConfiguration('max_steer'),
                'use_external_dynamic': 'false',
            }.items(),
        ),

        # 2) g-g-v 护盾：/drive_raw -> /drive（mux navigation 口）
        Node(
            package='slash_safety',
            executable='ggv_shield_node',
            name='ggv_shield_node',
            output='screen',
            parameters=[
                {
                    'drive_in_topic': LaunchConfiguration('raw_drive_topic'),
                    'drive_out_topic': LaunchConfiguration('out_drive_topic'),
                    'odom_topic': LaunchConfiguration('odom_topic'),
                    'debug_topic': '/ggv_shield/v_cap',
                    'wheelbase': ParameterValue(LaunchConfiguration('shield_wheelbase'), value_type=float),
                    'ay_max': ParameterValue(LaunchConfiguration('shield_ay_max'), value_type=float),
                    'v_max': ParameterValue(LaunchConfiguration('shield_v_max'), value_type=float),
                    'v_min': ParameterValue(LaunchConfiguration('shield_v_min'), value_type=float),
                    'steering_limit_rad': ParameterValue(
                        LaunchConfiguration('shield_steering_limit'), value_type=float),
                    'ay_safety_factor': ParameterValue(
                        LaunchConfiguration('shield_ay_safety_factor'), value_type=float),
                    'ax_accel_max': ParameterValue(
                        LaunchConfiguration('shield_ax_accel_max'), value_type=float),
                    'ax_brake_max': ParameterValue(
                        LaunchConfiguration('shield_ax_brake_max'), value_type=float),
                    'control_rate_hz': ParameterValue(
                        LaunchConfiguration('shield_control_rate_hz'), value_type=float),
                    'input_timeout_s': ParameterValue(
                        LaunchConfiguration('shield_input_timeout_s'), value_type=float),
                    'odom_timeout_s': ParameterValue(
                        LaunchConfiguration('shield_odom_timeout_s'), value_type=float),
                    'use_odom_speed_as_ref': ParameterValue(
                        LaunchConfiguration('use_odom_speed_as_ref'), value_type=bool),
                    'publish_debug': True,
                },
            ],
        ),
    ])
