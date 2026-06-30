#!/usr/bin/env python3
"""R2 reachability controller — live chain monitor.

One refreshed panel showing everything needed to see *why* the car moves or not:
  - /scan regional nearest distances (front / FL / FR / L / R)
  - /drive            : R2 (wall_following2) output  -> speed, steer
  - /ackermann_cmd    : ackermann_mux output         -> speed, steer
  - /commands/motor/speed : erpm to VESC
  - /commands/servo/position : servo to VESC
  - /sensors/core     : VESC voltage / rpm / fault
  - /rosout           : R2's reachability decision reason (no_reachable_arc ...)

Plus a one-line VERDICT that classifies the failure mode:
  R2 NOT PUBLISHING / R2 HOLDING speed=0 / MUX BLOCKED / AUTONOMY OK

QoS note: /scan is BEST_EFFORT, the command/sensor chain is RELIABLE.
A BEST_EFFORT subscriber receives from both, so we use it everywhere.

Run (same DDS env as the bringup!):
  source /opt/ros/foxy/setup.bash
  source ~/slash_ws/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI="<CycloneDDS><Domain><General><NetworkInterfaceAddress>eth0</NetworkInterfaceAddress></General></Domain></CycloneDDS>"
  python3 ~/slash_ws/r2_live_monitor.py
"""
import sys
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan
from vesc_msgs.msg import VescStateStamped
from rcl_interfaces.msg import Log

# Universal receiver: BEST_EFFORT subscribes fine to RELIABLE publishers too.
BE = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

STALE_SEC = 1.0


def _rad(lo_deg, hi_deg):
    return math.radians(lo_deg), math.radians(hi_deg)


# Regional bands (degrees, 0 = straight ahead, + = left, - = right).
BANDS = [
    ("F", _rad(-15, 15)),
    ("FL", _rad(15, 50)),
    ("FR", _rad(-50, -15)),
    ("L", _rad(50, 90)),
    ("R", _rad(-90, -50)),
]


class R2Monitor(Node):
    def __init__(self):
        super().__init__("r2_live_monitor")
        self.scan = None
        self.drive = None
        self.drive_t = None
        self.ack = None
        self.ack_t = None
        self.mot = None
        self.mot_t = None
        self.srv = None
        self.srv_t = None
        self.core = None
        self.core_t = None
        self.r2log = None
        self.r2log_t = None
        self.start = time.monotonic()
        self.prev_lines = 0

        self.create_subscription(LaserScan, "/scan", self._on_scan, BE)
        self.create_subscription(AckermannDriveStamped, "/drive", self._on_drive, BE)
        self.create_subscription(AckermannDriveStamped, "/ackermann_cmd", self._on_ack, BE)
        self.create_subscription(Float64, "/commands/motor/speed", self._on_mot, BE)
        self.create_subscription(Float64, "/commands/servo/position", self._on_srv, BE)
        self.create_subscription(VescStateStamped, "/sensors/core", self._on_core, BE)
        self.create_subscription(Log, "/rosout", self._on_log, BE)

        self.create_timer(0.1, self._render)
        self.get_logger().info(
            "R2 live monitor up. Watching /scan /drive /ackermann_cmd "
            "/commands/motor/speed /commands/servo/position /sensors/core /rosout"
        )

    # ---- callbacks --------------------------------------------------------
    def _on_scan(self, msg):
        self.scan = msg

    def _on_drive(self, msg):
        self.drive = msg
        self.drive_t = time.monotonic()

    def _on_ack(self, msg):
        self.ack = msg
        self.ack_t = time.monotonic()

    def _on_mot(self, msg):
        self.mot = msg.data
        self.mot_t = time.monotonic()

    def _on_srv(self, msg):
        self.srv = msg.data
        self.srv_t = time.monotonic()

    def _on_core(self, msg):
        self.core = msg.state
        self.core_t = time.monotonic()

    def _on_log(self, msg):
        m = msg.msg or ""
        if "Reachability" in m or "no_reachable" in m or "reachability" in m.lower():
            self.r2log = m.strip()
            self.r2log_t = time.monotonic()

    # ---- helpers ----------------------------------------------------------
    def _regions(self):
        s = self.scan
        if s is None:
            return None
        angles = s.angle_min + np.arange(len(s.ranges)) * s.angle_increment
        r = np.asarray(s.ranges, dtype=float)
        finite = np.isfinite(r) & (r > 0.05)
        out = []
        for _, (lo, hi) in BANDS:
            m = finite & (angles >= lo) & (angles <= hi)
            out.append(float(r[m].min()) if m.any() else float("inf"))
        return out  # [F, FL, FR, L, R]

    @staticmethod
    def _fmt_dist(v):
        return "--" if v is None or not math.isfinite(v) else f"{v:.2f}"

    @staticmethod
    def _age(t):
        return None if t is None else time.monotonic() - t

    def _verdict(self, ds, as_):
        d_age = self._age(self.drive_t)
        if d_age is None or d_age > STALE_SEC:
            return "R2 NOT PUBLISHING /drive  (battle_fast2 node not running?)"
        if ds <= 0.01:
            reason = self.r2log if (self.r2log_t and time.monotonic() - self.r2log_t < 2.0) else "speed=0"
            return f"R2 HOLDING  speed=0  ({reason})"
        if as_ <= 0.01 and ds > 0.01:
            return "MUX BLOCKED  /drive>0 but /ackermann_cmd=0  (check /teleop priority 100 vs /drive 10)"
        return f"AUTONOMY OK  drive={ds:.2f} -> ack={as_:.2f}  motor={self.mot or 0:.0f}erpm"

    # ---- render -----------------------------------------------------------
    def _render(self):
        regs = self._regions()
        if regs is not None:
            f, fl, fr, left, right = regs
            scan_min = min(regs)
        else:
            f = fl = fr = left = right = scan_min = None

        ds = self.drive.drive.speed if self.drive else 0.0
        dd = math.degrees(self.drive.drive.steering_angle) if self.drive else 0.0
        d_age = self._age(self.drive_t)
        d_state = (
            "NO PUB"
            if d_age is None
            else ("STALE" if d_age > STALE_SEC else "live")
        )

        a_age = self._age(self.ack_t)
        a_state = (
            "NO PUB" if a_age is None else ("STALE" if a_age > STALE_SEC else "live")
        )
        as_ = self.ack.drive.speed if self.ack else 0.0
        ad = math.degrees(self.ack.drive.steering_angle) if self.ack else 0.0

        mot = self.mot if self.mot is not None else None
        srv = self.srv if self.srv is not None else None

        volt = rpm = fault = None
        if self.core is not None:
            volt = getattr(self.core, "voltage_input", None)
            rpm = getattr(self.core, "speed", None)
            fault = getattr(self.core, "fault_code", None)

        r2log_disp = self.r2log if (self.r2log_t and time.monotonic() - self.r2log_t < 2.0) else "(none recent)"

        verdict = self._verdict(ds, as_)
        elapsed = time.monotonic() - self.start

        lines = [
            f"=== R2 live monitor   t={elapsed:6.1f}s   ===",
            f"VERDICT : {verdict}",
            f"scan(m) : F={self._fmt_dist(f)}  FL={self._fmt_dist(fl)}  FR={self._fmt_dist(fr)}  "
            f"L={self._fmt_dist(left)}  R={self._fmt_dist(right)}  min={self._fmt_dist(scan_min)}",
            f"R2 /drive        : speed={ds:6.3f}  steer={dd:+6.1f}deg   [{d_state}]",
            f"mux /ackermann_cmd : speed={as_:6.3f}  steer={ad:+6.1f}deg   [{a_state}]",
            f"VESC cmd  : motor={mot if mot is None else f'{mot:8.0f}erpm'}  "
            f"servo={srv if srv is None else f'{srv:.3f}'}",
            f"VESC core : {volt if volt is None else f'{volt:4.1f}V'}  "
            f"{rpm if rpm is None else f'{rpm:8.0f}rpm'}  "
            f"fault={fault if fault is None else int(fault)}",
            f"R2 /rosout: {r2log_disp}",
        ]

        out = "\n".join(lines)
        if sys.stdout.isatty():
            if self.prev_lines:
                sys.stdout.write(f"\033[{self.prev_lines}A")
            sys.stdout.write("\033[J")
            sys.stdout.write(out + "\n")
            self.prev_lines = len(lines)
        else:
            sys.stdout.write(out + "\n")
        sys.stdout.flush()


def main():
    rclpy.init()
    node = R2Monitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if sys.stdout.isatty():
            sys.stdout.write("\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
