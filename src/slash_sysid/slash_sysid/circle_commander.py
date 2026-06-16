#! /usr/bin/env python3
# coding=utf-8
"""
绕圆标定指令发生器（circle_commander）

目的：让车以**恒定转角 + 缓升速度**画圆，录一段 /sensors/imu + /odom + /commands/servo/position 的
rosbag，供 slash_sysid 的 identify_lateral 实测横向极限 ay_max / 有效轴距（W2）。

安全设计：
  - 速度从 0 缓升到 v_target（ramp_time 秒），不阶跃；到 hold_time 后自动降速停车并退出。
  - 全程速度 <= v_target（默认 1.2 m/s，保守）；转角固定（默认 0.25 rad）。
  - 指令默认发到 /drive（mux navigation 口，优先级 10）；手柄 /teleop(优先级100) 可随时接管/急停。
  - 收到 Ctrl-C 立即发停车再退出。

用法（先确保整车 bringup 在跑、mux 在、场地空旷）：
  ros2 run slash_sysid circle_commander --ros-args \
      -p steering_angle:=0.25 -p v_target:=1.2 -p ramp_time:=3.0 -p hold_time:=15.0
同时另开一个终端录包：
  ros2 bag record /odom /sensors/imu /commands/servo/position /commands/motor/speed /ackermann_cmd \
      -o circle_test
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from ackermann_msgs.msg import AckermannDriveStamped


class CircleCommander(Node):
    def __init__(self):
        super().__init__("circle_commander")
        params = [
            ("drive_topic", "/drive"),
            ("steering_angle", 0.25),   # rad，固定转角（画圆）
            ("v_target", 1.2),          # m/s，目标车速（保守）
            ("ramp_time", 3.0),         # s，0 -> v_target 缓升
            ("hold_time", 15.0),        # s，恒速保持时长
            ("brake_time", 2.0),        # s，结束缓降到 0
            ("rate_hz", 50.0),
            ("direction", 1.0),         # +1 左转 / -1 右转
        ]
        for n, d in params:
            self.declare_parameter(n, d)
        g = lambda n: self.get_parameter(n).value

        self.drive_topic = str(g("drive_topic"))
        self.steer = float(g("steering_angle")) * float(g("direction"))
        self.v_target = float(g("v_target"))
        self.ramp_t = max(0.1, float(g("ramp_time")))
        self.hold_t = float(g("hold_time"))
        self.brake_t = max(0.1, float(g("brake_time")))
        self.rate = max(1.0, float(g("rate_hz")))

        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, qos)
        self.t_start = self.get_clock().now().nanoseconds * 1e-9
        self.timer = self.create_timer(1.0 / self.rate, self._tick)
        self.get_logger().info(
            f"[circle_commander] 画圆标定开始：steer={self.steer:.3f}rad "
            f"v_target={self.v_target}m/s ramp={self.ramp_t}s hold={self.hold_t}s. "
            f"记得另开终端 ros2 bag record /odom /sensors/imu ...；手柄随时可接管。")

    def _profile(self, t):
        """返回该时刻目标速度（梯形剖面）。"""
        if t < self.ramp_t:
            return self.v_target * (t / self.ramp_t)
        if t < self.ramp_t + self.hold_t:
            return self.v_target
        td = t - self.ramp_t - self.hold_t
        if td < self.brake_t:
            return self.v_target * max(0.0, 1.0 - td / self.brake_t)
        return None  # 结束

    def _publish(self, steer, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed)
        self.pub.publish(msg)

    def _tick(self):
        t = self.get_clock().now().nanoseconds * 1e-9 - self.t_start
        v = self._profile(t)
        if v is None:
            self._publish(0.0, 0.0)
            self.get_logger().info("[circle_commander] 标定结束，已停车。可 Ctrl-C 退出并停止录包。")
            self.timer.cancel()
            return
        self._publish(self.steer, v)

    def stop(self):
        for _ in range(5):
            self._publish(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = CircleCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
