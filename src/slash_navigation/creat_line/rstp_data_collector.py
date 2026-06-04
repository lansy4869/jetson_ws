#!/usr/bin/env python3
# coding=utf-8
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import os
from datetime import datetime
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
import copy

class RSTPDataCollector(Node):
    def __init__(self):
        super().__init__('rstp_data_collector')

        # ====================== 参数 ======================
        self.declare_parameter('save_dir', '/home/jetson/rstp_datasets')
        self.declare_parameter('sequence_length', 20)      # τ_{1:L} 的 horizon（和扩散模型一致）
        self.declare_parameter('save_every_n_laps', 1)     # 每几圈保存一次（防止内存爆炸）
        self.save_dir = self.get_parameter('save_dir').value
        self.seq_len = int(self.get_parameter('sequence_length').value)
        self.save_every = int(self.get_parameter('save_every_n_laps').value)

        os.makedirs(self.save_dir, exist_ok=True)
        self.dataset = []           # 当前正在采集的 sequence
        self.sequences_saved = 0
        self.lap_count = 0
        self.last_odom = None

        # QoS 和你节点一致
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20
        )

        # 订阅你的现有话题
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos)          # 或你的 debug_scan_topic
        self.create_subscription(Path, '/battle_fast2/local_ref', self.ref_callback, 10)  # 你刚才加的
        self.create_subscription(Path, '/raceline', self.raceline_callback, 1)

        self.get_logger().info(f"✅ RSTP 离线数据采集器启动！保存路径: {self.save_dir}")
        self.get_logger().info(f"   每 {self.seq_len} 步生成一条专家轨迹 (s_t, o_t, m_t, τ*)")

    def odom_callback(self, msg):
        self.last_odom = msg

    def scan_callback(self, msg):
        if self.last_odom is None:
            return
        # 记录一条数据
        data_point = {
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            's_t': {                                      # 当前状态
                'x': msg.pose.pose.position.x,            # 注意：这里用 odom 的 pose
                'y': msg.pose.pose.position.y,
                'yaw': self._quat_to_yaw(msg.pose.pose.orientation),
                'v': float(msg.twist.twist.linear.x)
            },
            'o_t': np.array(msg.ranges).astype(np.float32),   # 完整 LiDAR（或后面改成 dis_90）
            'm_t': self.last_raceline if hasattr(self, 'last_raceline') else None,
            'tau': None                                   # 等 ref 到来再填
        }
        self.dataset.append(data_point)

        # 达到 sequence_length 就保存一条完整轨迹
        if len(self.dataset) >= self.seq_len:
            self._save_sequence()

    def ref_callback(self, msg):
        """收到专家局部轨迹（你的 build_local_ref 返回的 ref）"""
        if not self.dataset:
            return
        # 把最新一条数据的 tau 填上（世界坐标下的轨迹点）
        traj = []
        for pose in msg.poses:
            traj.append([pose.pose.position.x, pose.pose.position.y])
        self.dataset[-1]['tau'] = np.array(traj).astype(np.float32)   # shape (L, 2)

    def raceline_callback(self, msg):
        """缓存全局 Raceline 作为 m_t"""
        self.last_raceline = msg  # 可后续采样局部段

    def _quat_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _save_sequence(self):
        if len(self.dataset) < self.seq_len:
            return
        # 只保留完整 sequence
        seq = self.dataset[:self.seq_len]
        self.dataset = self.dataset[self.seq_len:]   # 滑动窗口，可改成不重叠

        filename = f"rstp_seq_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.sequences_saved:06d}.npz"
        save_path = os.path.join(self.save_dir, filename)

        # 转成 npz（扩散训练最友好格式）
        np.savez_compressed(save_path,
                            states=[d['s_t'] for d in seq],
                            obs=[d['o_t'] for d in seq],
                            maps=[d.get('m_t') for d in seq],   # 可后续处理
                            expert_traj=[d['tau'] for d in seq if d['tau'] is not None])

        self.sequences_saved += 1
        self.get_logger().info(f"💾 已保存第 {self.sequences_saved} 条训练序列 → {filename}")

        # 每几圈打印统计（可选）
        if self.sequences_saved % 10 == 0:
            self.get_logger().info(f"🎉 当前已采集 {self.sequences_saved} 条专家轨迹！")

    def destroy_node(self):
        if self.dataset:
            self._save_sequence()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RSTPDataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 数据采集已停止")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()