from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _f(name):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def generate_launch_description():
    """绕圆标定指令发生器（配合 ros2 bag record 采集横向标定数据）。"""
    return LaunchDescription([
        DeclareLaunchArgument('drive_topic', default_value='/drive'),
        DeclareLaunchArgument('steering_angle', default_value='0.25'),
        DeclareLaunchArgument('v_target', default_value='1.2'),
        DeclareLaunchArgument('ramp_time', default_value='3.0'),
        DeclareLaunchArgument('hold_time', default_value='15.0'),
        DeclareLaunchArgument('direction', default_value='1.0'),
        Node(
            package='slash_sysid',
            executable='circle_commander',
            name='circle_commander',
            output='screen',
            parameters=[{
                'drive_topic': LaunchConfiguration('drive_topic'),
                'steering_angle': _f('steering_angle'),
                'v_target': _f('v_target'),
                'ramp_time': _f('ramp_time'),
                'hold_time': _f('hold_time'),
                'direction': _f('direction'),
            }],
        ),
    ])
