#! /usr/bin/env python3
# coding=utf-8
"""
g-g-v 安全护盾节点（创新点 3 的安全壳模块）

作用：坐在「规划器输出」与「ackermann_mux 的 navigation 口」之间，
把任意上游策略（battle_fast2 反应式 / 未来的 BC 学生网络）的 (steering, speed) 指令
投影回**本车实测可行域**，再发给 mux。这是与具体策略正交的、可解释的物理安全层。

接线（默认）：
    上游规划器  --(drive_in: /drive_raw)-->  [ggv_shield]  --(drive_out: /drive)-->  ackermann_mux(navigation, prio 10)
    /odom (VESC 轮式里程计) --> 当前车速，用于纵向变化率限制

护盾做四件事（全部参数化）：
  1) 转向限幅：|delta| <= steering_limit_rad（实测 servo 行程对应 ~±0.34 rad）
  2) 过弯限速（g-g-v 横轴）：kappa = tan(|delta|)/L，v <= sqrt(ay_max*ay_safety_factor / kappa)
  3) 纵向变化率限制（g-g-v 纵轴）：dv 受实测 ax_accel_max / ax_brake_max 约束
  4) 失效安全：超时收不到上游指令或里程计 → 输出停车（speed=0）

默认参数用 sysid 实测值（identified_params.json）：
  ax_accel_max=6.35, ax_brake_max=6.66 (取正幅值), 横向 ay_max 实测前先用保守假设。

第一性原理：
  摩擦椭圆  (ax/ax_max)^2 + (ay/ay_max)^2 <= 1
  稳态过弯  ay = v^2 * kappa,  kappa = tan(delta)/L
  => 过弯最大稳态车速  v_max(kappa) = sqrt(ay_max / kappa)
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


class GgvShieldNode(Node):
    def __init__(self):
        super().__init__("ggv_shield_node")

        # === 参数声明（全部可由 launch / yaml 注入） ===
        params = [
            # 接线
            ("drive_in_topic", "/drive_raw"),     # 订阅上游规划器原始指令
            ("drive_out_topic", "/drive"),        # 发给 mux 的 navigation 口
            ("odom_topic", "/odom"),              # 取当前车速（VESC 轮式里程计）
            ("debug_topic", "/ggv_shield/v_cap"), # 调试：当前过弯限速
            # 车辆几何与实测极限（默认 = sysid 实测）
            ("wheelbase", 0.25),                  # m，vesc.yaml 实测
            ("ay_max", 9.81),                     # m/s^2，横向极限（绕圆 bag 标定前用 mu*g 假设）
            ("ay_safety_factor", 0.6),            # 横向安全系数（首跑保守，逐步放开到 1.0）
            ("ax_accel_max", 6.35),               # m/s^2，实测加速能力
            ("ax_brake_max", 6.66),               # m/s^2，实测制动能力（正幅值）
            ("steering_limit_rad", 0.34),         # rad，实测 servo 行程对应转角
            ("v_max", 2.0),                       # m/s，全局限速上限
            ("v_min", 0.0),                       # m/s，竞速不倒车，下限 0
            # 运行
            ("control_rate_hz", 50.0),            # 看门狗 / 兜底发布频率
            ("input_timeout_s", 0.3),             # 超过此时间没收到上游指令 → 停车
            ("odom_timeout_s", 0.5),              # 超过此时间没收到里程计 → 用上一次指令速度做变化率参考
            ("use_odom_speed_as_ref", True),      # True: dv 以实测车速为基准；False: 以上一次输出为基准
            ("publish_debug", True),
        ]
        for name, default in params:
            self.declare_parameter(name, default)
        g = lambda n: self.get_parameter(n).value

        self.drive_in_topic = str(g("drive_in_topic"))
        self.drive_out_topic = str(g("drive_out_topic"))
        self.odom_topic = str(g("odom_topic"))
        self.debug_topic = str(g("debug_topic"))
        self.L = float(g("wheelbase"))
        self.ay_max = float(g("ay_max"))
        self.ay_factor = float(g("ay_safety_factor"))
        self.ax_accel = float(g("ax_accel_max"))
        self.ax_brake = abs(float(g("ax_brake_max")))
        self.steer_limit = float(g("steering_limit_rad"))
        self.v_max = float(g("v_max"))
        self.v_min = float(g("v_min"))
        self.control_rate = max(1.0, float(g("control_rate_hz")))
        self.input_timeout = float(g("input_timeout_s"))
        self.odom_timeout = float(g("odom_timeout_s"))
        self.use_odom_ref = bool(g("use_odom_speed_as_ref"))
        self.publish_debug = bool(g("publish_debug"))

        # 运行时状态
        self.last_out_speed = 0.0
        self.measured_speed = 0.0
        self.last_input_t = None      # 上游指令最近到达时刻 (s)
        self.last_odom_t = None       # 里程计最近到达时刻 (s)
        self.last_pub_t = None        # 上次输出时刻（算 dt）
        self.last_steer = 0.0
        self.have_input_once = False

        # QoS：控制指令用 RELIABLE depth=1；里程计 BEST_EFFORT
        ctrl_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_out_topic, ctrl_qos)
        self.debug_pub = self.create_publisher(Float32, self.debug_topic, 10) if self.publish_debug else None
        self.drive_sub = self.create_subscription(
            AckermannDriveStamped, self.drive_in_topic, self._on_drive_in, ctrl_qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._on_odom, sensor_qos
        )
        self.watchdog = self.create_timer(1.0 / self.control_rate, self._watchdog)

        self.get_logger().info(
            f"[ggv_shield] {self.drive_in_topic} -> {self.drive_out_topic} | "
            f"L={self.L} ay_max={self.ay_max}*{self.ay_factor} "
            f"ax=[+{self.ax_accel},-{self.ax_brake}] steer<=±{self.steer_limit} v<=({self.v_min},{self.v_max})"
        )

    # ---------- 工具 ----------
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _curvature_speed_cap(self, steer):
        """过弯限速：v <= sqrt(ay_eff / kappa)，kappa = tan(|delta|)/L。"""
        kappa = abs(math.tan(steer)) / max(self.L, 1e-6)
        if kappa < 1e-6:
            return self.v_max
        ay_eff = max(0.0, self.ay_max * self.ay_factor)
        v_curve = math.sqrt(ay_eff / kappa)
        return min(self.v_max, v_curve)

    def _rate_limit_speed(self, v_des, dt):
        """纵向变化率限制：dv 落在 [-ax_brake*dt, +ax_accel*dt]。"""
        now = self._now()
        has_fresh_odom = (
            self.last_odom_t is not None
            and (now - self.last_odom_t) <= self.odom_timeout
        )
        ref = self.measured_speed if self.use_odom_ref and has_fresh_odom else self.last_out_speed
        dv = v_des - ref
        dv = max(-self.ax_brake * dt, min(self.ax_accel * dt, dv))
        return ref + dv

    def _shield(self, steer_req, v_req, dt):
        """把 (steer_req, v_req) 投影回可行域，返回 (steer, v, v_cap)。"""
        steer = max(-self.steer_limit, min(self.steer_limit, float(steer_req)))
        v_cap = self._curvature_speed_cap(steer)
        v_des = max(self.v_min, min(v_cap, float(v_req)))
        v_out = self._rate_limit_speed(v_des, dt)
        v_out = max(self.v_min, min(v_cap, v_out))  # 变化率限制后再夹一次上限
        return steer, v_out, v_cap

    def _publish(self, steer, speed, stamp_header=None):
        msg = AckermannDriveStamped()
        if stamp_header is not None:
            msg.header = stamp_header
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)
        self.last_out_speed = float(speed)
        self.last_steer = float(steer)
        self.last_pub_t = self._now()

    # ---------- 回调 ----------
    def _on_odom(self, msg):
        self.measured_speed = float(msg.twist.twist.linear.x)
        self.last_odom_t = self._now()

    def _on_drive_in(self, msg):
        now = self._now()
        dt = 1.0 / self.control_rate if self.last_pub_t is None else max(1e-3, min(0.2, now - self.last_pub_t))
        steer, v_out, v_cap = self._shield(msg.drive.steering_angle, msg.drive.speed, dt)
        self.last_input_t = now
        self.have_input_once = True
        self._publish(steer, v_out, stamp_header=msg.header)
        if self.debug_pub is not None:
            self.debug_pub.publish(Float32(data=float(v_cap)))

    def _watchdog(self):
        """没有有效上游指令时兜底停车（按制动极限平滑减速到 0）。"""
        now = self._now()
        if self.last_input_t is not None and (now - self.last_input_t) <= self.input_timeout:
            return  # 上游还在正常发指令，事件驱动已处理
        if not self.have_input_once:
            # 启动阶段可能还没有 /scan，planner 因而尚未发布 /drive_raw。
            # 此时保持安全停车即可，不按“运行中断流”刷屏报警。
            self._publish(0.0, 0.0)
            self.get_logger().info(
                f"[ggv_shield] 等待上游指令 {self.drive_in_topic}，当前保持停车",
                throttle_duration_sec=10.0,
            )
            return
        # 失效：平滑刹停
        dt = 1.0 / self.control_rate if self.last_pub_t is None else max(1e-3, min(0.2, now - self.last_pub_t))
        ref = self.measured_speed if (self.use_odom_ref and self.last_odom_t is not None
                                      and (now - self.last_odom_t) <= self.odom_timeout) else self.last_out_speed
        v_out = max(0.0, ref - self.ax_brake * dt)
        self._publish(self.last_steer * 0.0, v_out)  # 转向回中、按制动极限减速
        if abs(v_out) < 1e-3:
            self.get_logger().warn("[ggv_shield] 上游指令超时，已安全停车（转向回中）", throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = GgvShieldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
