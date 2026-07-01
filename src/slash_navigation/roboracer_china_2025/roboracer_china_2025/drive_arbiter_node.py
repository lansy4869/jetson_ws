#! /usr/bin/env python3

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from .drive_arbiter_core import (
    ArbiterConfig,
    DriveCommand,
    SafetyDiagnostics,
    ShieldConfig,
    SourceHealth,
    arbitrate,
)


def _strip_leading_slashes(value):
    return value.lstrip("/")


class DriveArbiterNode(Node):
    def __init__(self):
        super().__init__("drive_arbiter")

        self.declare_parameter("mpc_drive_topic", "/mpc/drive_nominal")
        self.declare_parameter("reactive_drive_topic", "/battle_fast2/drive_reactive")
        self.declare_parameter("odom_topic", "/pf/pose/odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("output_frame", "base_link")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("command_timeout_s", 0.2)
        self.declare_parameter("odom_timeout_s", 0.3)
        self.declare_parameter("max_steer", 0.412)
        self.declare_parameter("max_speed", 2.0)
        self.declare_parameter("min_speed", 0.0)
        self.declare_parameter("prefer_reactive_on_mpc_timeout", True)
        self.declare_parameter("enable_safety_shield", True)
        self.declare_parameter("front_clearance_topic", "/battle_fast2/front_clearance_m")
        self.declare_parameter("risk_min_margin_topic", "/battle_fast2/risk_min_margin_m")
        self.declare_parameter(
            "reactive_speed_limit_topic",
            "/battle_fast2/reactive_speed_limit_mps",
        )
        self.declare_parameter("shield_diag_timeout_s", 0.2)
        self.declare_parameter("shield_yellow_margin_m", 0.55)
        self.declare_parameter("shield_orange_margin_m", 0.35)
        self.declare_parameter("shield_red_margin_m", 0.18)
        self.declare_parameter("shield_black_clearance_m", 0.30)
        self.declare_parameter("shield_orange_blend", 0.55)

        self.mpc_drive_topic = self.get_parameter("mpc_drive_topic").value
        self.reactive_drive_topic = self.get_parameter("reactive_drive_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.drive_topic = self.get_parameter("drive_topic").value
        self.front_clearance_topic = self.get_parameter("front_clearance_topic").value
        self.risk_min_margin_topic = self.get_parameter("risk_min_margin_topic").value
        self.reactive_speed_limit_topic = self.get_parameter(
            "reactive_speed_limit_topic"
        ).value
        self.global_frame = _strip_leading_slashes(
            str(self.get_parameter("global_frame").value)
        )
        self.output_frame = str(self.get_parameter("output_frame").value)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.config = ArbiterConfig(
            max_steer=float(self.get_parameter("max_steer").value),
            max_speed=float(self.get_parameter("max_speed").value),
            min_speed=float(self.get_parameter("min_speed").value),
            command_timeout_s=float(self.get_parameter("command_timeout_s").value),
            odom_timeout_s=float(self.get_parameter("odom_timeout_s").value),
            prefer_reactive_on_mpc_timeout=bool(
                self.get_parameter("prefer_reactive_on_mpc_timeout").value
            ),
        )
        self.shield_config = ShieldConfig(
            enabled=bool(self.get_parameter("enable_safety_shield").value),
            diag_timeout_s=float(self.get_parameter("shield_diag_timeout_s").value),
            yellow_margin_m=float(self.get_parameter("shield_yellow_margin_m").value),
            orange_margin_m=float(self.get_parameter("shield_orange_margin_m").value),
            red_margin_m=float(self.get_parameter("shield_red_margin_m").value),
            black_clearance_m=float(
                self.get_parameter("shield_black_clearance_m").value
            ),
            orange_blend=float(self.get_parameter("shield_orange_blend").value),
        )

        self.mpc_command = None
        self.mpc_health = SourceHealth(stamp_s=None, valid=False, reason="mpc missing")
        self.reactive_command = None
        self.reactive_health = SourceHealth(
            stamp_s=None,
            valid=False,
            reason="reactive missing",
        )
        self.odom_health = SourceHealth(stamp_s=None, valid=False, reason="odom missing")
        self.front_clearance = None
        self.risk_min_margin = None
        self.reactive_speed_limit = None
        self.front_clearance_stamp_s = None
        self.risk_min_margin_stamp_s = None
        self.reactive_speed_limit_stamp_s = None
        self.safety_diagnostics = None
        self.safety_health = SourceHealth(
            stamp_s=None,
            valid=False,
            reason="safety diagnostics missing",
        )
        self.last_source = None

        self.create_subscription(
            AckermannDriveStamped,
            self.mpc_drive_topic,
            self._mpc_callback,
            25,
        )
        self.create_subscription(
            AckermannDriveStamped,
            self.reactive_drive_topic,
            self._reactive_callback,
            25,
        )
        self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_callback,
            25,
        )
        self.create_subscription(
            Float32,
            self.front_clearance_topic,
            self._front_clearance_callback,
            25,
        )
        self.create_subscription(
            Float32,
            self.risk_min_margin_topic,
            self._risk_min_margin_callback,
            25,
        )
        self.create_subscription(
            Float32,
            self.reactive_speed_limit_topic,
            self._reactive_speed_limit_callback,
            25,
        )
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            self.drive_topic,
            25,
        )

        timer_period = 1.0 / publish_rate_hz if publish_rate_hz > 0.0 else 1.0 / 30.0
        self.create_timer(timer_period, self._timer_callback)

        self.get_logger().info(
            "Drive arbiter ready: mpc=%s reactive=%s odom=%s output=%s"
            % (
                self.mpc_drive_topic,
                self.reactive_drive_topic,
                self.odom_topic,
                self.drive_topic,
            )
        )

    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _mpc_callback(self, msg):
        self.mpc_command = DriveCommand(
            steering_angle=float(msg.drive.steering_angle),
            speed=float(msg.drive.speed),
        )
        self.mpc_health = SourceHealth(stamp_s=self._now_s(), valid=True)

    def _reactive_callback(self, msg):
        self.reactive_command = DriveCommand(
            steering_angle=float(msg.drive.steering_angle),
            speed=float(msg.drive.speed),
        )
        self.reactive_health = SourceHealth(stamp_s=self._now_s(), valid=True)

    def _front_clearance_callback(self, msg):
        self.front_clearance = float(msg.data)
        self.front_clearance_stamp_s = self._now_s()
        self._refresh_safety_diagnostics()

    def _risk_min_margin_callback(self, msg):
        self.risk_min_margin = float(msg.data)
        self.risk_min_margin_stamp_s = self._now_s()
        self._refresh_safety_diagnostics()

    def _reactive_speed_limit_callback(self, msg):
        self.reactive_speed_limit = float(msg.data)
        self.reactive_speed_limit_stamp_s = self._now_s()
        self._refresh_safety_diagnostics()

    def _refresh_safety_diagnostics(self):
        if (
            self.front_clearance is None
            or self.risk_min_margin is None
            or self.reactive_speed_limit is None
        ):
            self.safety_health = SourceHealth(
                stamp_s=None,
                valid=False,
                reason="safety diagnostics missing",
            )
            return

        latest_complete_stamp = min(
            self.front_clearance_stamp_s,
            self.risk_min_margin_stamp_s,
            self.reactive_speed_limit_stamp_s,
        )
        self.safety_diagnostics = SafetyDiagnostics(
            front_clearance=float(self.front_clearance),
            risk_min_margin=float(self.risk_min_margin),
            reactive_speed_limit=float(self.reactive_speed_limit),
        )
        self.safety_health = SourceHealth(stamp_s=latest_complete_stamp, valid=True)

    def _odom_callback(self, msg):
        frame_id = _strip_leading_slashes(msg.header.frame_id)
        if frame_id != self.global_frame:
            self.odom_health = SourceHealth(
                stamp_s=self._now_s(),
                valid=False,
                reason="odom frame %s expected %s" % (frame_id, self.global_frame),
            )
            return

        self.odom_health = SourceHealth(stamp_s=self._now_s(), valid=True)

    def _timer_callback(self):
        decision = arbitrate(
            now_s=self._now_s(),
            mpc_command=self.mpc_command,
            mpc_health=self.mpc_health,
            reactive_command=self.reactive_command,
            reactive_health=self.reactive_health,
            odom_health=self.odom_health,
            config=self.config,
            safety_diagnostics=self.safety_diagnostics,
            safety_health=self.safety_health,
            shield_config=self.shield_config,
        )
        self.drive_publisher.publish(self._to_msg(decision.command))

        if decision.source != self.last_source:
            self.get_logger().warn(
                "Drive source switched to %s: %s" % (decision.source, decision.reason)
            )
            self.last_source = decision.source

    def _to_msg(self, command):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.output_frame
        msg.drive.steering_angle = float(command.steering_angle)
        msg.drive.speed = float(command.speed)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = DriveArbiterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
