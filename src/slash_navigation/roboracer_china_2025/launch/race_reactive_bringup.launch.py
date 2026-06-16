#! /usr/bin/env python3
# coding=utf-8
"""
W1 整车部署底座 + 反应式跑通（一条命令拉起）。

链路：
    MID-360 ──/livox/lidar──> [linefit 地面分割] ──/segmentation/obstacle──>
        [pointcloud_to_laserscan] ──/scan──> [battle_fast2 反应式规划器] ──/drive──>
        [ackermann_mux(navigation)] ──> vesc

同时 f1tenth_stack bringup 起 vesc / 里程计(/odom) / 手柄(/teleop, 可随时接管) / mux /
base_link→livox_frame 静态 TF。

设计：
  - 默认 run_planner=true，即 W1 单独跑通（反应式 → /drive，保守速度）。
  - 需要叠加 W2 安全壳 / W3 多层感知时，用 run_planner:=false 把本 launch 当作
    “纯底座（雷达+感知+底盘+/scan+/odom+mux）”，再在其上跑 race_with_shield / race_3d，
    避免两个规划器同时抢 /drive。

各组件可单独开关（lidar / seg / p2l / chassis / planner），便于排障。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    livox_share = get_package_share_directory('livox_ros_driver2')
    linefit_share = get_package_share_directory('linefit_ground_segmentation_ros')
    p2l_share = get_package_share_directory('pointcloud_to_laserscan')
    stack_share = get_package_share_directory('f1tenth_stack')
    planner_share = get_package_share_directory('roboracer_china_2025')

    args = [
        DeclareLaunchArgument('use_lidar', default_value='true'),
        DeclareLaunchArgument('use_segmentation', default_value='true'),
        DeclareLaunchArgument('use_pointcloud_to_laserscan', default_value='true'),
        DeclareLaunchArgument('use_chassis', default_value='true'),
        DeclareLaunchArgument('run_planner', default_value='true',
                              description='true=W1 反应式直跑；false=只起底座，留给 W2/W3 叠加'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('drive_topic', default_value='/drive'),
        DeclareLaunchArgument('speed_scale', default_value='0.6'),
        DeclareLaunchArgument('max_speed', default_value='2.0'),
        DeclareLaunchArgument('max_steer', default_value='0.34'),
        DeclareLaunchArgument('verbose', default_value='false'),
    ]

    # 1) MID-360 驱动 -> /livox/lidar
    lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(livox_share, 'launch', 'msg_MID360_launch.py')),
        condition=IfCondition(LaunchConfiguration('use_lidar')),
    )

    # 2) linefit 地面分割 -> /segmentation/obstacle, /segmentation/ground
    segmentation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(linefit_share, 'launch', 'segmentation.launch.py')),
        condition=IfCondition(LaunchConfiguration('use_segmentation')),
    )

    # 3) 障碍点云 -> 2D /scan
    p2l = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(p2l_share, 'launch', 'pointcloud_to_laserscan_launch.py')),
        condition=IfCondition(LaunchConfiguration('use_pointcloud_to_laserscan')),
    )

    # 4) 整车底盘 bringup（vesc / 里程计 / 手柄 / mux / 静态 TF）
    chassis = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(stack_share, 'launch', 'bringup_launch.py')),
        condition=IfCondition(LaunchConfiguration('use_chassis')),
    )

    # 5) 反应式规划器（保守速度）-> /drive
    planner = GroupAction(
        condition=IfCondition(LaunchConfiguration('run_planner')),
        actions=[
            Node(
                package='roboracer_china_2025',
                executable='battle_fast2_node',
                name='battle_fast2_node',
                output='screen',
                parameters=[{
                    'scan_topic': LaunchConfiguration('scan_topic'),
                    'drive_topic': LaunchConfiguration('drive_topic'),
                    'speed_scale': ParameterValue(LaunchConfiguration('speed_scale'), value_type=float),
                    'max_speed': ParameterValue(LaunchConfiguration('max_speed'), value_type=float),
                    'max_steer': ParameterValue(LaunchConfiguration('max_steer'), value_type=float),
                    'verbose': ParameterValue(LaunchConfiguration('verbose'), value_type=bool),
                }],
            ),
        ],
    )
    # planner_share 仅用于确保依赖包已安装（launch 文件路径自包含）
    _ = planner_share

    return LaunchDescription(args + [lidar, segmentation, p2l, chassis, planner])
