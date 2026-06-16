import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """W7 部署全栈（W3 感知 + W4 测速 + 学生策略 + W2 护盾）：

        /livox/lidar --[multilayer_scan]--> /scan_3d
        /scan_3d --[obstacle_tracker]--> /perception/opponent_state (+ dynamic flag)
        /scan_3d + /odom + opponent_state --[student_policy_node]--> /drive_raw
        /drive_raw --[ggv_shield]--> /drive --> ackermann_mux(navigation)

    学生超时/低置信/后端缺失 → 节点内反应式兜底；g-g-v 护盾在 ggv_shield 统一兜底。

    前置：W1 底座在跑（雷达/p2l/odom/mux），建议
        ros2 launch roboracer_china_2025 race_reactive_bringup.launch.py run_planner:=false
    """
    ml_share = get_package_share_directory('multilayer_scan')
    shield_share = get_package_share_directory('slash_safety')
    tracker_share = get_package_share_directory('obstacle_tracker')

    return LaunchDescription([
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('input_topic', default_value='/livox/lidar'),
        DeclareLaunchArgument('fused_scan_topic', default_value='/scan_3d'),
        DeclareLaunchArgument('opponent_topic', default_value='/perception/opponent_state'),
        DeclareLaunchArgument('raw_drive_topic', default_value='/drive_raw'),
        DeclareLaunchArgument('out_drive_topic', default_value='/drive'),
        DeclareLaunchArgument('model_path', default_value='',
                              description='学生 .onnx / .pt 路径；空=纯反应式兜底'),
        DeclareLaunchArgument('backend', default_value='onnx',
                              description='onnx / trt / torch'),
        DeclareLaunchArgument('ml_params',
                              default_value=os.path.join(ml_share, 'config', 'multilayer_scan.yaml')),
        DeclareLaunchArgument('shield_params',
                              default_value=os.path.join(shield_share, 'config', 'ggv_shield.yaml')),

        # 1) 三维多层感知 -> /scan_3d
        Node(
            package='multilayer_scan', executable='multilayer_scan_node',
            name='multilayer_scan_node', output='screen',
            parameters=[
                LaunchConfiguration('ml_params'),
                {'input_topic': LaunchConfiguration('input_topic'),
                 'fused_scan_topic': LaunchConfiguration('fused_scan_topic')},
            ],
        ),

        # 2) 运动学测速 -> opponent_state（学生对手通道）
        Node(
            package='obstacle_tracker', executable='obstacle_tracker_node',
            name='obstacle_tracker_node', output='screen',
            parameters=[
                os.path.join(tracker_share, 'config', 'obstacle_tracker.yaml'),
                {'scan_topic': LaunchConfiguration('fused_scan_topic'),
                 'odom_topic': LaunchConfiguration('odom_topic')},
            ],
        ),

        # 3) 学生策略：/scan_3d + /odom + opponent_state -> /drive_raw
        Node(
            package='slash_distill', executable='student_policy_node',
            name='student_policy_node', output='screen',
            parameters=[{
                'scan_topics': [LaunchConfiguration('fused_scan_topic')],
                'odom_topic': LaunchConfiguration('odom_topic'),
                'opponent_topic': LaunchConfiguration('opponent_topic'),
                'drive_out_topic': LaunchConfiguration('raw_drive_topic'),
                'model_path': LaunchConfiguration('model_path'),
                'backend': LaunchConfiguration('backend'),
            }],
        ),

        # 4) g-g-v 安全护盾：/drive_raw -> /drive
        Node(
            package='slash_safety', executable='ggv_shield_node',
            name='ggv_shield_node', output='screen',
            parameters=[
                LaunchConfiguration('shield_params'),
                {'drive_in_topic': LaunchConfiguration('raw_drive_topic'),
                 'drive_out_topic': LaunchConfiguration('out_drive_topic'),
                 'odom_topic': LaunchConfiguration('odom_topic')},
            ],
        ),
    ])
