#! /usr/bin/env python3
# coding=utf-8
"""
三维多层感知节点（multilayer_scan）—— 毕设创新点 1：3D 多层反应式竞速

动机：MID-360 是三维雷达，但现有管线（pointcloud_to_laserscan）只取**一层**高度切片
(min_height=-1.0, max_height=0.10) 压成单条 2D /scan，把全部高度结构都丢了。
本节点把点云按**多个高度带**投影成多条 LaserScan，从而区分
「低矮障碍 / 车体高度障碍 / 悬空物」，再融合成给反应式规划器用的 /scan_3d。

输入选择（重要，贴合本车实际）：默认订阅 **/livox/lidar**（MID-360 PointCloud2 原始点云）。
现有 /scan 由同一份三维点云经地面分割和 pointcloud_to_laserscan 生成，所以多层节点同源、不引入新依赖。
地面靠**高度带下边界**避开（雷达装在 base_link 上方 ~0.11m，livox_frame 内 z≈0 即雷达水平面，
地面约在 z≈-0.11 以下；把 low 带下界放在地面之上即可）。
若以后启用 linefit，可改 input_topic:=/segmentation/obstacle，并把 low 带下界放更低。

设计意图：把单层 /scan 升级成多带，区分「低矮 / 车体高度 / 悬空」障碍；阻挡带逐角度取最近距离融合，
悬空层 `high` 默认 **不阻挡**（blocking=false），避免门楣/桌沿等高处回波制造假墙（这正是相对单层 /scan 的改进）。
band 高度为 RViz 可调起点，需在实车按现象标定。

第一性原理（投影几何）：点 (x,y,z) 在 livox_frame 内，
  range = hypot(x,y)，angle = atan2(y,x)，bin = round((angle-angle_min)/angle_increment)
按 z 落入某高度带；每带每角度取最近 range；融合 = 阻挡带逐角度 min。

输出：
  - 每层一条调试 scan： <layer_scan_prefix><name>（如 /perception/scan_layer_body）
  - 一条融合 scan（给规划器）： fused_scan_topic（默认 /scan_3d）
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan, PointCloud2


# ROS PointField datatype -> numpy
_PF_TO_NP = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
             5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def pointcloud2_to_xyz(msg):
    """用结构化 dtype 直接从原始 buffer 取 x/y/z（对 Livox PointXYZRTL 的额外字段鲁棒）。"""
    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.name in ("x", "y", "z") and f.datatype in _PF_TO_NP:
            names.append(f.name)
            formats.append(_PF_TO_NP[f.datatype])
            offsets.append(f.offset)
    if set(names) != {"x", "y", "z"}:
        return None
    dt = np.dtype({"names": names, "formats": formats,
                   "offsets": offsets, "itemsize": msg.point_step})
    n = msg.width * msg.height
    arr = np.frombuffer(bytes(msg.data), dtype=dt, count=n)
    xyz = np.empty((arr.shape[0], 3), dtype=np.float64)
    xyz[:, 0] = arr["x"]
    xyz[:, 1] = arr["y"]
    xyz[:, 2] = arr["z"]
    return xyz


class MultiLayerScanNode(Node):
    def __init__(self):
        super().__init__("multilayer_scan_node")

        params = [
            # 接线
            ("input_topic", "/livox/lidar"),  # MID-360 PointCloud2；启用 linefit 后可改 /segmentation/obstacle
            ("fused_scan_topic", "/scan_3d"),
            ("layer_scan_prefix", "/perception/scan_layer_"),
            ("frame_id", "livox_frame"),
            ("publish_layers", True),       # 是否发布每层调试 scan（RViz 验证用）
            # 扫描几何（与现有 /scan 对齐，便于 drop-in 替换）
            ("angle_min", -3.14159),
            ("angle_max", 3.14159),
            ("angle_increment", 0.0043),
            ("range_min", 0.1),
            ("range_max", 20.0),
            ("use_inf", True),              # 无回波填 inf（反应式前端用 nan_to_num 处理）
            # 高度带（livox_frame z，单位 m）：三组并行数组（RViz 可调起点）
            # low/body 阻挡、high 悬空不阻挡；下界须在地面之上（地面约 z≈-0.11 以下）
            ("band_names", ["low", "body", "high"]),
            ("band_floor", [-0.08, -0.05, 0.10]),
            ("band_ceil", [-0.05, 0.10, 0.50]),
            ("band_blocking", [True, True, False]),  # 进融合 scan 的层（high 默认不阻挡）
            # 自车体过滤盒（|x|<ego_box_x 且 |y|<ego_box_y 的点丢弃）
            ("ego_box_x", 0.30),
            ("ego_box_y", 0.20),
        ]
        for n, d in params:
            self.declare_parameter(n, d)
        g = lambda n: self.get_parameter(n).value

        self.fused_topic = str(g("fused_scan_topic"))
        self.layer_prefix = str(g("layer_scan_prefix"))
        self.frame_id = str(g("frame_id"))
        self.publish_layers = bool(g("publish_layers"))
        self.angle_min = float(g("angle_min"))
        self.angle_max = float(g("angle_max"))
        self.angle_inc = float(g("angle_increment"))
        self.range_min = float(g("range_min"))
        self.range_max = float(g("range_max"))
        self.use_inf = bool(g("use_inf"))
        self.b_names = [str(x) for x in g("band_names")]
        self.b_floor = [float(x) for x in g("band_floor")]
        self.b_ceil = [float(x) for x in g("band_ceil")]
        self.b_block = [bool(x) for x in g("band_blocking")]
        self.ego_x = float(g("ego_box_x"))
        self.ego_y = float(g("ego_box_y"))

        nb = len(self.b_names)
        if not (len(self.b_floor) == len(self.b_ceil) == len(self.b_block) == nb):
            raise ValueError("band_names/floor/ceil/blocking 长度必须一致")
        self.n_beams = int(round((self.angle_max - self.angle_min) / self.angle_inc))
        self.fill_val = float("inf") if self.use_inf else (self.range_max + 1.0)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5,
        )
        self.fused_pub = self.create_publisher(LaserScan, self.fused_topic, sensor_qos)
        self.layer_pubs = {}
        if self.publish_layers:
            for name in self.b_names:
                self.layer_pubs[name] = self.create_publisher(
                    LaserScan, self.layer_prefix + name, sensor_qos)
        self.sub = self.create_subscription(
            PointCloud2, str(g("input_topic")), self._on_cloud, sensor_qos)

        self.get_logger().info(
            f"[multilayer_scan] {g('input_topic')} -> {self.fused_topic} | "
            f"beams={self.n_beams} bands={list(zip(self.b_names, self.b_floor, self.b_ceil, self.b_block))} "
            f"ego_box=({self.ego_x},{self.ego_y})")

    def _make_scan(self, ranges, header):
        msg = LaserScan()
        msg.header = header
        msg.header.frame_id = self.frame_id
        msg.angle_min = self.angle_min
        msg.angle_max = self.angle_max
        msg.angle_increment = self.angle_inc
        msg.range_min = self.range_min
        msg.range_max = self.range_max
        msg.ranges = ranges.astype(np.float32).tolist()
        return msg

    def _on_cloud(self, msg):
        xyz = pointcloud2_to_xyz(msg)
        if xyz is None or xyz.shape[0] == 0:
            return
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = np.hypot(x, y)
        ang = np.arctan2(y, x)

        valid = np.isfinite(r) & (r >= self.range_min) & (r <= self.range_max)
        valid &= ~((np.abs(x) < self.ego_x) & (np.abs(y) < self.ego_y))  # 自车体盒
        idx = np.round((ang - self.angle_min) / self.angle_inc).astype(np.int64)
        idx = np.mod(idx, self.n_beams)  # 360° 周期：边界角(±π)正确回绕，不丢波束

        x, y, z, r, idx = x[valid], y[valid], z[valid], r[valid], idx[valid]

        fused = np.full(self.n_beams, self.fill_val, dtype=np.float64)
        for name, lo, hi, block in zip(self.b_names, self.b_floor, self.b_ceil, self.b_block):
            m = (z >= lo) & (z < hi)
            layer = np.full(self.n_beams, self.fill_val, dtype=np.float64)
            if m.any():
                np.minimum.at(layer, idx[m], r[m])
            if block:
                fused = np.minimum(fused, layer)
            if self.publish_layers:
                self.layer_pubs[name].publish(self._make_scan(layer, msg.header))

        self.fused_pub.publish(self._make_scan(fused, msg.header))


def main(args=None):
    rclpy.init(args=args)
    node = MultiLayerScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
