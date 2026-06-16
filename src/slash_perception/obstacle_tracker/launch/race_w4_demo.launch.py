#! /usr/bin/env python3
# coding=utf-8
"""
W4 全栈演示链（W1底座 + W3 多层感知 + W4 运动学测速 + W2 安全壳）：

    /livox/lidar --[multilayer_scan]--> /scan_3d
        ├─[obstacle_tracker]--> /perception/dynamic_obstacle + /perception/opponent_state + markers
        └─[battle_fast2 (scan:=/scan_3d, use_external_dynamic:=true)]--> /drive_raw
              --[ggv_shield]--> /drive --> ackermann_mux(navigation)

前置：整车 bringup（雷达/p2l/底盘/odom/mux）已在跑（见 W4 使用指南）。

用途：单车演示「避静态障碍 + 单个动态障碍跟随/避让」。动态判据来自运动学测速（创新点2），
不再用 intensity。RViz 看 /perception/obstacle_markers：动态障碍变红并带速度箭头。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    ml_share = get_package_share_directory('multilayer_scan')
    tracker_share = get_package_share_directory('obstacle_tracker')
    planner_share = get_package_share_directory('roboracer_china_2025')
    shield_share = get_package_share_directory('slash_safety')

    fused_scan = LaunchConfiguration('fused_scan_topic')
    dyn_topic = LaunchConfiguration('dynamic_flag_topic')

    return LaunchDescription([
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('input_topic', default_value='/livox/lidar'),
        DeclareLaunchArgument('fused_scan_topic', default_value='/scan_3d'),
        DeclareLaunchArgument('dynamic_flag_topic', default_value='/perception/dynamic_obstacle'),
        DeclareLaunchArgument('raw_drive_topic', default_value='/drive_raw'),
        DeclareLaunchArgument('out_drive_topic', default_value='/drive'),
        DeclareLaunchArgument('speed_scale', default_value='0.6'),
        DeclareLaunchArgument('ml_params',
                              default_value=os.path.join(ml_share, 'config', 'multilayer_scan.yaml')),
        DeclareLaunchArgument('tracker_params',
                              default_value=os.path.join(tracker_share, 'config', 'obstacle_tracker.yaml')),
        DeclareLaunchArgument('shield_params',
                              default_value=os.path.join(shield_share, 'config', 'ggv_shield.yaml')),

        # 1) W3 三维多层感知 -> /scan_3d
        Node(
            package='multilayer_scan',
            executable='multilayer_scan_node',
            name='multilayer_scan_node',
            output='screen',
            parameters=[
                LaunchConfiguration('ml_params'),
                {'input_topic': LaunchConfiguration('input_topic'),
                 'fused_scan_topic': fused_scan},
            ],
        ),

        # 2) W4 障碍运动学跟踪/测速 -> dynamic flag + opponent_state
        Node(
            package='obstacle_tracker',
            executable='obstacle_tracker_node',
            name='obstacle_tracker_node',
            output='screen',
            parameters=[
                LaunchConfiguration('tracker_params'),
                {'scan_topic': fused_scan,
                 'odom_topic': LaunchConfiguration('odom_topic'),
                 'dynamic_flag_topic': dyn_topic},
            ],
        ),

        # 3) 反应式规划器：吃 /scan_3d + 运动学动态判据，发 /drive_raw
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(planner_share, 'launch', 'battle_fast2.launch.py')),
            launch_arguments={
                'scan_topic': fused_scan,
                'drive_topic': LaunchConfiguration('raw_drive_topic'),
                'speed_scale': LaunchConfiguration('speed_scale'),
                'use_external_dynamic': 'true',
                'dynamic_flag_topic': dyn_topic,
            }.items(),
        ),

        # 4) W2 g-g-v 安全护盾：/drive_raw -> /drive
        Node(
            package='slash_safety',
            executable='ggv_shield_node',
            name='ggv_shield_node',
            output='screen',
            parameters=[
                LaunchConfiguration('shield_params'),
                {
                    'drive_in_topic': LaunchConfiguration('raw_drive_topic'),
                    'drive_out_topic': LaunchConfiguration('out_drive_topic'),
                    'odom_topic': LaunchConfiguration('odom_topic'),
                },
            ],
        ),
    ])
