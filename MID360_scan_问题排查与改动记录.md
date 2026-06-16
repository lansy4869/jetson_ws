# MID360 `/scan` 问题排查与改动记录

记录时间：2026-06-13

## 1. 问题现象

启动 MID360 雷达后执行：

```bash
ros2 topic hz /scan
```

输出：

```text
WARNING: topic [/scan] does not appear to be published yet
```

也就是说导航节点默认需要的 `/scan` 没有发布。

## 2. 排查结论

现场 ROS 图里能看到 MID360 驱动已经启动，并发布：

```text
/livox/lidar [sensor_msgs/msg/PointCloud2]
/livox/imu   [sensor_msgs/msg/Imu]
```

`/livox/lidar` 能 echo 到 PointCloud2 点云，说明雷达、网络和 Livox 驱动本身是通的。

真正的问题是：MID360 驱动只发布 3D 点云 `/livox/lidar`，不会自动发布 2D LaserScan `/scan`。需要额外启动 `pointcloud_to_laserscan_node` 把点云转换成 `/scan`。

同时发现原来的转换 launch 配置订阅的是：

```text
/segmentation/obstacle
```

但当前只启动雷达时并没有这个 topic，所以转换节点即使启动，也不会从 `/livox/lidar` 生成 `/scan`。

## 3. 已做的运行时处理

临时启动了正式转换节点：

```bash
ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node --ros-args \
  -r __node:=pointcloud_to_laserscan \
  -r cloud_in:=/livox/lidar \
  -r scan:=/scan \
  -p min_height:=-1.0 \
  -p max_height:=0.2 \
  -p range_min:=0.1 \
  -p range_max:=20.0 \
  -p angle_min:=-3.14159 \
  -p angle_max:=3.14159 \
  -p angle_increment:=0.0043 \
  -p use_inf:=true
```

清理了之前测试用的 `/scan_probe` 节点，避免 topic 列表混淆。

## 4. 已修改的文件

### 4.1 `rviz_MID360_launch.py`

文件：

```text
src/slash_hardware/livox_ros_driver2/src/launch/rviz_MID360_launch.py
```

改动：

在 MID360 驱动和 RViz 之外，新增 `pointcloud_to_laserscan_node`，自动执行：

```text
/livox/lidar -> /scan
```

关键参数：

```python
remappings=[
    ('cloud_in', '/livox/lidar'),
    ('scan', '/scan'),
]
parameters=[{
    'target_frame': frame_id,
    'transform_tolerance': 0.01,
    'min_height': -1.0,
    'max_height': 0.2,
    'angle_min': -3.14159,
    'angle_max': 3.14159,
    'angle_increment': 0.0043,
    'scan_time': 1.0 / publish_freq,
    'range_min': 0.1,
    'range_max': 20.0,
    'use_inf': True,
    'inf_epsilon': 1.0,
}]
```

以后启动：

```bash
ros2 launch livox_ros_driver2 rviz_MID360_launch.py
```

会同时发布 `/livox/lidar` 和 `/scan`。

### 4.2 `msg_MID360_launch.py`

文件：

```text
src/slash_hardware/livox_ros_driver2/src/launch/msg_MID360_launch.py
```

改动：

同样新增 `pointcloud_to_laserscan_node`，不打开 RViz 时也自动生成 `/scan`。

以后启动：

```bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

也会同时发布 `/livox/lidar` 和 `/scan`。

### 4.3 `pointcloud_to_laserscan_launch.py`

文件：

```text
src/slash_perception/pointcloud_to_laserscan/launch/pointcloud_to_laserscan_launch.py
```

改动：

把输入从不存在的：

```text
/segmentation/obstacle
```

改为当前 MID360 实际发布的：

```text
/livox/lidar
```

并把转换参数调整为实测可用配置：

```python
'max_height': 0.2,
'scan_time': 0.1,
'range_min': 0.1,
'range_max': 20.0,
```

## 5. 验证结果

语法检查通过：

```bash
python3 -m py_compile \
  src/slash_hardware/livox_ros_driver2/src/launch/rviz_MID360_launch.py \
  src/slash_hardware/livox_ros_driver2/src/launch/msg_MID360_launch.py \
  src/slash_perception/pointcloud_to_laserscan/launch/pointcloud_to_laserscan_launch.py
```

ROS topic 确认：

```text
/livox/lidar [sensor_msgs/msg/PointCloud2]
/scan        [sensor_msgs/msg/LaserScan]
```

`/scan` 发布者：

```text
Node name: pointcloud_to_laserscan
Topic type: sensor_msgs/msg/LaserScan
Publisher count: 1
```

频率验证：

```bash
ros2 topic hz /scan
```

结果稳定在约：

```text
average rate: 10.0 Hz
```

与 MID360 当前 `publish_freq = 10.0` 一致。

## 6. 后续推荐启动方式

带 RViz：

```bash
source /opt/ros/foxy/setup.bash
source ~/slash_ws/install/setup.bash
ros2 launch livox_ros_driver2 rviz_MID360_launch.py
```

不带 RViz：

```bash
source /opt/ros/foxy/setup.bash
source ~/slash_ws/install/setup.bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

验证：

```bash
ros2 topic hz /scan
```

启动导航节点时，`battle_fast2_node` 默认订阅 `/scan`，所以不需要额外改 `scan_topic`。

## 7. 注意事项

`pointcloud_to_laserscan_node` 是按需订阅的：只有 `/scan` 有订阅者时，它才会订阅 `/livox/lidar` 并开始转换。日志里出现下面内容是正常的：

```text
Got a subscriber to laserscan, starting pointcloud subscriber
No subscribers to laserscan, shutting down pointcloud subscriber
```

如果下次 `/scan` 又没有数据，优先检查：

```bash
ros2 topic list -t | grep livox
ros2 topic echo /livox/lidar
ros2 topic info /scan
ros2 topic hz /scan
```

若 `/livox/lidar` 没有数据，则问题在雷达驱动、网口 IP 或 MID360 连接；若 `/livox/lidar` 有数据但 `/scan` 没有，则检查 `pointcloud_to_laserscan_node` 是否启动。
