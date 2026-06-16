#! /usr/bin/env python3
# coding=utf-8
"""W7 部署节点：BC/DAgger 学生策略 → /drive_raw（再经 ggv_shield 进 mux）。

接线：
  /scan_3d（或 low/body/high 多层）+ /odom + /perception/opponent_state
     → obs.build_obs → ONNX(onnxruntime, 可挂 TensorRT EP) / torch → (steer, speed)
     → /drive_raw → [slash_safety/ggv_shield] → /drive → ackermann_mux(navigation)

安全设计（与 mpc_execplan 的 off/steer_only/full 思想一致）：
  - 推理后端不可用 / 模型缺失 → 始终走反应式兜底（节点仍能开车，降级不致命）
  - 学生输出 NaN / 与反应式速度差过大 → 该帧切反应式
  - 传感器超时 → 不发指令，由下游 ggv_shield 看门狗刹停
  - g-g-v 物理护盾在下游 ggv_shield 节点统一做（本节点只发 /drive_raw）
"""
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from ackermann_msgs.msg import AckermannDriveStamped

from ..common import obs as obsmod
from ..common.reactive import ReactiveController
from ..models.student_cnn import denormalize_action


class _Inference:
    """统一推理后端：onnx(onnxruntime) / torch / none。none 时由节点走反应式。"""

    def __init__(self, logger, model_path: str, backend: str):
        self.kind = "none"
        self.sess = None
        self.policy = None
        self.steer_max = None
        self.v_max = None
        if not model_path:
            logger.warn("[student] 未配置 model_path → 纯反应式兜底模式")
            return
        # 载入 meta（若有）
        import os
        meta_npz = os.path.splitext(model_path)[0] + "_meta.npz"
        if os.path.isfile(meta_npz):
            m = np.load(meta_npz)
            self.steer_max = float(m["steer_max"]); self.v_max = float(m["v_max"])

        if backend in ("onnx", "trt") and model_path.endswith(".onnx"):
            try:
                import onnxruntime as ort
                providers = []
                avail = ort.get_available_providers()
                if backend == "trt" and "TensorrtExecutionProvider" in avail:
                    providers.append("TensorrtExecutionProvider")
                if "CUDAExecutionProvider" in avail:
                    providers.append("CUDAExecutionProvider")
                providers.append("CPUExecutionProvider")
                self.sess = ort.InferenceSession(model_path, providers=providers)
                self.kind = "onnx"
                logger.info(f"[student] onnxruntime 后端 providers={self.sess.get_providers()}")
            except Exception as e:
                logger.error(f"[student] onnxruntime 载入失败 → 反应式兜底：{e}")
        elif model_path.endswith(".pt"):
            try:
                from ..models.policy import TorchStudentPolicy
                self.policy = TorchStudentPolicy(model_path, device="cpu")
                self.steer_max = self.policy.steer_max; self.v_max = self.policy.v_max
                self.kind = "torch"
                logger.info("[student] torch 后端")
            except Exception as e:
                logger.error(f"[student] torch 载入失败 → 反应式兜底：{e}")

    def available(self) -> bool:
        return self.kind != "none"

    def infer(self, lidar: np.ndarray, scalar: np.ndarray):
        """返回原始 (steer, speed)。失败返回 None。"""
        try:
            if self.kind == "onnx":
                out = self.sess.run(None, {"lidar": lidar[None].astype(np.float32),
                                           "scalar": scalar[None].astype(np.float32)})[0][0]
                return denormalize_action(float(out[0]), float(out[1]),
                                          self.steer_max, self.v_max)
            if self.kind == "torch":
                return self.policy.act(lidar, scalar)
        except Exception:
            return None
        return None


class StudentPolicyNode(Node):
    def __init__(self):
        super().__init__("student_policy_node")
        p = [
            ("scan_topics", ["/scan_3d"]),     # 1层=融合scan；3层=低/体/高层
            ("odom_topic", "/odom"),
            ("opponent_topic", "/perception/opponent_state"),
            ("drive_out_topic", "/drive_raw"),
            ("model_path", ""),                # .onnx 或 .pt；空=纯反应式
            ("backend", "onnx"),               # onnx / trt / torch
            ("control_rate_hz", 50.0),
            ("sensor_timeout_s", 0.3),
            ("use_fallback", True),
            ("fallback_speed_jump", 1.5),
            # 观测维度（须与训练一致；有 meta_npz 时 steer/v 用 meta）
            ("n_beams", 108),
            ("fov_rad", 4.7),
            ("range_max", 20.0),
            ("use_opponent", True),
            ("steer_max", 0.34),
            ("v_max", 2.0),
        ]
        for n, d in p:
            self.declare_parameter(n, d)
        g = lambda n: self.get_parameter(n).value

        self.scan_topics = list(g("scan_topics"))
        self.odom_topic = str(g("odom_topic"))
        self.opp_topic = str(g("opponent_topic"))
        self.drive_out = str(g("drive_out_topic"))
        self.rate = max(1.0, float(g("control_rate_hz")))
        self.timeout = float(g("sensor_timeout_s"))
        self.use_fallback = bool(g("use_fallback"))
        self.fb_jump = float(g("fallback_speed_jump"))

        self.obs_cfg = obsmod.ObsConfig(
            n_beams=int(g("n_beams")), n_layers=len(self.scan_topics),
            fov_rad=float(g("fov_rad")), range_max=float(g("range_max")),
            use_opponent=bool(g("use_opponent")), v_max=float(g("v_max")))

        self.infer = _Inference(self.get_logger(), str(g("model_path")), str(g("backend")))
        if self.infer.v_max is None:
            self.infer.steer_max = float(g("steer_max")); self.infer.v_max = float(g("v_max"))
        self.reactive = ReactiveController(steer_limit=float(g("steer_max")),
                                           speed_max=float(g("v_max")))

        # 运行状态
        self.scans = [None] * len(self.scan_topics)
        self.scan_t = [None] * len(self.scan_topics)
        self.ego_speed = 0.0
        self.opp = None
        self._last_log = 0.0

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        ctrl_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                              history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        for i, t in enumerate(self.scan_topics):
            self.create_subscription(LaserScan, t, self._make_scan_cb(i), qos)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, qos)
        self.create_subscription(Float32MultiArray, self.opp_topic, self._on_opp, qos)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_out, ctrl_qos)
        self.timer = self.create_timer(1.0 / self.rate, self._tick)

        self.get_logger().info(
            f"[student] scans={self.scan_topics} -> {self.drive_out} | "
            f"backend={self.infer.kind} n_beams={self.obs_cfg.n_beams} "
            f"layers={self.obs_cfg.n_layers} fallback={self.use_fallback}")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _make_scan_cb(self, i):
        def cb(msg):
            self.scans[i] = msg
            self.scan_t[i] = self._now()
        return cb

    def _on_odom(self, msg):
        self.ego_speed = float(msg.twist.twist.linear.x)

    def _on_opp(self, msg):
        if msg.data:
            self.opp = np.asarray(msg.data, dtype=np.float64)

    def _fresh(self):
        now = self._now()
        return all(t is not None and (now - t) <= self.timeout for t in self.scan_t)

    def _layers(self):
        layers = []
        for s in self.scans:
            if s is None:
                return None
            layers.append((np.asarray(s.ranges, dtype=np.float64),
                           float(s.angle_min), float(s.angle_increment)))
        return layers

    def _tick(self):
        if not self._fresh():
            return  # 传感器超时：不发指令，下游 ggv_shield 看门狗刹停
        layers = self._layers()
        if layers is None:
            return
        body = layers[len(layers) // 2]  # 用"体层/融合层"做反应式与兜底
        lidar, scalar = obsmod.build_obs(layers, self.ego_speed, self.opp, self.obs_cfg)

        used_fallback = False
        out = self.infer.infer(lidar, scalar) if self.infer.available() else None
        if out is None:
            steer, speed = self.reactive.compute(body[0], body[1], body[2])
            used_fallback = True
        else:
            steer, speed = out
            if self.use_fallback:
                r_steer, r_speed = self.reactive.compute(body[0], body[1], body[2])
                if (not np.isfinite(steer)) or (not np.isfinite(speed)) or \
                        abs(speed - r_speed) > self.fb_jump:
                    steer, speed = r_steer, r_speed
                    used_fallback = True

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

        now = self._now()
        if now - self._last_log > 2.0:
            self._last_log = now
            self.get_logger().info(
                f"[student] steer={steer:.3f} speed={speed:.2f} "
                f"{'(FALLBACK)' if used_fallback else f'({self.infer.kind})'}",
                throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = StudentPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
