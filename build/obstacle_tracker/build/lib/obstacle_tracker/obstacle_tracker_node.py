#! /usr/bin/env python3
# coding=utf-8
"""
障碍多帧运动学跟踪 + 测速节点（obstacle_tracker）—— 毕设创新点 2

动机：原 battle_fast2 用 LiDAR **intensity 均值差 > 5** 判动态障碍。MID-360 的反射率(0–255)
与 2D 雷达 intensity **不可比**，这条判据上实车不可移植（见 skill 硬约束 4）。本节点改用
**纯运动学**：对 LaserScan 做聚类 → 多帧数据关联 → 估计每个障碍的速度（接近/横向），
据此判定「动态障碍」并产出 opponent_state，喂给反应式规划器（替换 intensity 判据）
以及后续创新点 4 的 BC 学生网络。

第一性原理（为什么要做 ego 运动补偿）：
  雷达在车体系(livox_frame)里观测。车一旦前进，**静止**障碍的距离也在变小，
  帧间看起来就是“在动”。所以测到的帧间速度 v_meas 含自车运动分量，必须扣除：
      v_ground ≈ v_meas + v_ego + ω × p
  其中 p=(x,y) 为障碍在车体系坐标，v_ego=(vx,vy)、ω=(0,0,wz) 来自 /odom。
  ω × p = (-wz·y, wz·x)。补偿后静止障碍 v_ground≈0，真动态障碍才保留真实速度。

输入：
  - LaserScan（默认 /scan_3d，可改 /scan）；
  - Odometry（默认 /odom，VESC 轮式里程计）用于 ego 运动补偿（可关）。

输出：
  - /perception/dynamic_obstacle (std_msgs/Bool)：前向扇区内是否存在动态障碍（给 battle_fast2）。
  - /perception/opponent_state (std_msgs/Float32MultiArray)：选中的最相关前向障碍状态，
        布局 = [valid, range, bearing, x, y, vx, vy, speed, range_rate, is_dynamic]
        （全可测量，供创新点 4 学生观测；无障碍时 valid=0、其余 0）。
  - /perception/obstacle_markers (visualization_msgs/MarkerArray)：RViz 可视化（框+速度箭头+文字）。
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class Track:
    """单个障碍的跟踪状态（车体系 livox_frame）。"""
    __slots__ = ("id", "x", "y", "vx", "vy", "rng", "rng_rate",
                 "age", "missed", "speed_ema", "matched")

    def __init__(self, tid, x, y, rng):
        self.id = tid
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.rng = rng
        self.rng_rate = 0.0
        self.age = 1
        self.missed = 0
        self.speed_ema = 0.0
        self.matched = True


class ObstacleTrackerNode(Node):
    def __init__(self):
        super().__init__("obstacle_tracker_node")

        params = [
            # 接线
            ("scan_topic", "/scan_3d"),
            ("odom_topic", "/odom"),
            ("dynamic_flag_topic", "/perception/dynamic_obstacle"),
            ("opponent_state_topic", "/perception/opponent_state"),
            ("marker_topic", "/perception/obstacle_markers"),
            ("frame_id", "livox_frame"),
            ("publish_markers", True),
            # 前向扇区与有效距离
            ("front_fov_deg", 120.0),     # 只在 ±fov/2 的前向扇区找障碍
            ("range_min", 0.15),
            ("range_max", 8.0),
            # 聚类
            ("cluster_gap_m", 0.30),      # 相邻波束 range 跳变 > 此值则断簇
            ("max_cluster_width_m", 1.2), # 线宽超过此值视为墙/大结构，丢弃
            ("min_cluster_points", 2),
            # 数据关联与跟踪
            ("assoc_dist_m", 0.6),        # 检测中心与已有 track 的最大关联距离
            ("max_missed", 6),            # 连续丢失帧数超过则删除 track
            ("min_age_for_dynamic", 3),   # track 至少存活几帧才允许判动态
            ("vel_ema_alpha", 0.5),       # 速度 EMA 平滑系数
            ("dynamic_speed_thresh", 0.35),  # m/s，ego 补偿后速度超过即判动态
            # ego 运动补偿
            ("use_ego_compensation", True),
            ("odom_timeout_s", 0.5),
        ]
        for n, d in params:
            self.declare_parameter(n, d)
        g = lambda n: self.get_parameter(n).value

        self.frame_id = str(g("frame_id"))
        self.publish_markers = bool(g("publish_markers"))
        self.front_fov = math.radians(float(g("front_fov_deg")))
        self.range_min = float(g("range_min"))
        self.range_max = float(g("range_max"))
        self.cluster_gap = float(g("cluster_gap_m"))
        self.max_cluster_width = float(g("max_cluster_width_m"))
        self.min_cluster_pts = int(g("min_cluster_points"))
        self.assoc_dist = float(g("assoc_dist_m"))
        self.max_missed = int(g("max_missed"))
        self.min_age_dyn = int(g("min_age_for_dynamic"))
        self.alpha = float(g("vel_ema_alpha"))
        self.dyn_thresh = float(g("dynamic_speed_thresh"))
        self.use_ego = bool(g("use_ego_compensation"))
        self.odom_timeout = float(g("odom_timeout_s"))

        # 运行时状态
        self.tracks = []
        self._next_id = 0
        self.last_t = None
        self.ego_vx = 0.0
        self.ego_vy = 0.0
        self.ego_wz = 0.0
        self.last_odom_t = None

        sensor_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=QoSHistoryPolicy.KEEP_LAST, depth=5)

        self.dyn_pub = self.create_publisher(Bool, str(g("dynamic_flag_topic")), 10)
        self.opp_pub = self.create_publisher(Float32MultiArray, str(g("opponent_state_topic")), 10)
        self.marker_pub = (self.create_publisher(MarkerArray, str(g("marker_topic")), 10)
                           if self.publish_markers else None)
        self.scan_sub = self.create_subscription(
            LaserScan, str(g("scan_topic")), self._on_scan, sensor_qos)
        if self.use_ego:
            self.odom_sub = self.create_subscription(
                Odometry, str(g("odom_topic")), self._on_odom, sensor_qos)

        self.get_logger().info(
            f"[obstacle_tracker] scan={g('scan_topic')} odom={g('odom_topic') if self.use_ego else 'OFF'} "
            f"-> {g('dynamic_flag_topic')} + opponent_state | "
            f"fov=±{math.degrees(self.front_fov)/2:.0f}° dyn_thresh={self.dyn_thresh}m/s ego_comp={self.use_ego}")

    # ---------------- 回调 ----------------
    def _on_odom(self, msg):
        self.ego_vx = float(msg.twist.twist.linear.x)
        self.ego_vy = float(msg.twist.twist.linear.y)
        self.ego_wz = float(msg.twist.twist.angular.z)
        self.last_odom_t = self._now()

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _stamp_t(self, header):
        t = header.stamp.sec + header.stamp.nanosec * 1e-9
        return t if t > 0.0 else self._now()

    # ---------------- 聚类 ----------------
    def _cluster(self, msg):
        """在前向扇区内把 LaserScan 聚成紧凑障碍簇，返回 [(x,y,range,bearing), ...]。"""
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        n = len(ranges)
        if n == 0:
            return []
        idx = np.arange(n)
        ang = msg.angle_min + idx * msg.angle_increment
        finite = np.isfinite(ranges) & (ranges >= self.range_min) & (ranges <= self.range_max)
        in_fov = np.abs(np.arctan2(np.sin(ang), np.cos(ang))) <= (self.front_fov / 2.0)
        good = finite & in_fov

        clusters = []
        cur = []
        prev_i = None
        prev_r = None
        for i in range(n):
            if not good[i]:
                if cur:
                    clusters.append(cur); cur = []
                prev_i = None
                continue
            r = ranges[i]
            if prev_i is not None and (i - prev_i) <= 2 and abs(r - prev_r) <= self.cluster_gap:
                cur.append(i)
            else:
                if cur:
                    clusters.append(cur)
                cur = [i]
            prev_i = i
            prev_r = r
        if cur:
            clusters.append(cur)

        out = []
        for c in clusters:
            if len(c) < self.min_cluster_pts:
                continue
            rr = ranges[c]
            aa = ang[c]
            xs = rr * np.cos(aa)
            ys = rr * np.sin(aa)
            width = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
            if width > self.max_cluster_width:
                continue  # 太宽 -> 墙/大结构，不当作可跟踪障碍
            cx, cy = float(xs.mean()), float(ys.mean())
            out.append((cx, cy, float(np.hypot(cx, cy)), float(math.atan2(cy, cx))))
        return out

    # ---------------- 关联 + 更新 ----------------
    def _on_scan(self, msg):
        t = self._stamp_t(msg.header)
        dt = 0.0 if self.last_t is None else max(1e-3, min(0.5, t - self.last_t))
        self.last_t = t

        dets = self._cluster(msg)
        for tr in self.tracks:
            tr.matched = False

        # 贪心最近邻关联
        used = [False] * len(dets)
        for tr in self.tracks:
            best_j, best_d = -1, self.assoc_dist
            for j, (dx, dy, _, _) in enumerate(dets):
                if used[j]:
                    continue
                d = math.hypot(dx - tr.x, dy - tr.y)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0:
                used[best_j] = True
                self._update_track(tr, dets[best_j], dt)
            else:
                tr.missed += 1
                tr.matched = False

        # 未关联的检测 -> 新建 track
        for j, det in enumerate(dets):
            if not used[j]:
                tr = Track(self._next_id, det[0], det[1], det[2])
                self._next_id += 1
                self.tracks.append(tr)

        # 删除长期丢失的 track
        self.tracks = [tr for tr in self.tracks if tr.missed <= self.max_missed]

        self._publish(msg.header)

    def _update_track(self, tr, det, dt):
        dx, dy, rng, _ = det
        if dt > 1e-3:
            vx_meas = (dx - tr.x) / dt
            vy_meas = (dy - tr.y) / dt
            rr_meas = (rng - tr.rng) / dt
        else:
            vx_meas = vy_meas = rr_meas = 0.0

        # ego 运动补偿：v_ground = v_meas + v_ego + ω × p
        if self.use_ego and self.last_odom_t is not None \
                and (self._now() - self.last_odom_t) <= self.odom_timeout:
            vx_g = vx_meas + self.ego_vx - self.ego_wz * dy
            vy_g = vy_meas + self.ego_vy + self.ego_wz * dx
        else:
            vx_g, vy_g = vx_meas, vy_meas

        a = self.alpha
        tr.vx = a * vx_g + (1 - a) * tr.vx
        tr.vy = a * vy_g + (1 - a) * tr.vy
        tr.rng_rate = a * rr_meas + (1 - a) * tr.rng_rate
        tr.x, tr.y, tr.rng = dx, dy, rng
        tr.age += 1
        tr.missed = 0
        tr.matched = True
        speed = math.hypot(tr.vx, tr.vy)
        tr.speed_ema = a * speed + (1 - a) * tr.speed_ema

    def _is_dynamic(self, tr):
        return tr.age >= self.min_age_dyn and tr.speed_ema > self.dyn_thresh

    # ---------------- 输出 ----------------
    def _publish(self, header):
        any_dynamic = False
        # 选最相关的前向障碍：优先动态、再取最近
        best = None
        best_key = None
        for tr in self.tracks:
            if not tr.matched:
                continue
            dyn = self._is_dynamic(tr)
            any_dynamic = any_dynamic or dyn
            key = (0 if dyn else 1, tr.rng)  # 动态优先，其次最近
            if best_key is None or key < best_key:
                best_key, best = key, tr

        self.dyn_pub.publish(Bool(data=bool(any_dynamic)))

        opp = Float32MultiArray()
        if best is not None:
            opp.data = [1.0, float(best.rng), float(math.atan2(best.y, best.x)),
                        float(best.x), float(best.y), float(best.vx), float(best.vy),
                        float(math.hypot(best.vx, best.vy)), float(best.rng_rate),
                        1.0 if self._is_dynamic(best) else 0.0]
        else:
            opp.data = [0.0] * 10
        self.opp_pub.publish(opp)

        if self.marker_pub is not None:
            self._publish_markers(header)

    def _publish_markers(self, header):
        arr = MarkerArray()
        # 先发一个 DELETEALL 清旧
        clr = Marker()
        clr.header.frame_id = self.frame_id
        clr.action = Marker.DELETEALL
        arr.markers.append(clr)
        for tr in self.tracks:
            if not tr.matched:
                continue
            dyn = self._is_dynamic(tr)
            box = Marker()
            box.header.frame_id = self.frame_id
            box.header.stamp = header.stamp
            box.ns = "obstacle"
            box.id = tr.id * 3
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = tr.x
            box.pose.position.y = tr.y
            box.pose.position.z = 0.0
            box.pose.orientation.w = 1.0
            box.scale.x = box.scale.y = 0.25
            box.scale.z = 0.25
            box.color.a = 0.8
            box.color.r = 1.0 if dyn else 0.1
            box.color.g = 0.2 if dyn else 0.9
            box.color.b = 0.1
            arr.markers.append(box)

            arrow = Marker()
            arrow.header.frame_id = self.frame_id
            arrow.header.stamp = header.stamp
            arrow.ns = "vel"
            arrow.id = tr.id * 3 + 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.scale.x = 0.04
            arrow.scale.y = 0.08
            arrow.scale.z = 0.0
            arrow.color.a = 1.0
            arrow.color.r = 1.0
            arrow.color.g = 1.0
            arrow.color.b = 0.0
            arrow.points.append(Point(x=tr.x, y=tr.y, z=0.0))
            arrow.points.append(Point(x=tr.x + tr.vx, y=tr.y + tr.vy, z=0.0))
            arr.markers.append(arrow)

            txt = Marker()
            txt.header.frame_id = self.frame_id
            txt.header.stamp = header.stamp
            txt.ns = "speed"
            txt.id = tr.id * 3 + 2
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = tr.x
            txt.pose.position.y = tr.y
            txt.pose.position.z = 0.35
            txt.scale.z = 0.18
            txt.color.a = 1.0
            txt.color.r = txt.color.g = txt.color.b = 1.0
            txt.text = f"{tr.speed_ema:.2f}m/s{' DYN' if dyn else ''}"
            arr.markers.append(txt)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
