#!/usr/bin/env python3
# coding=utf-8
"""
BattleVehicleNode —— 高速无人赛车全栈控制节点
集成激光前端感知、全局-局部融合规划、MPC+PP混合控制、模式状态机

启动方式：
    python3 battle_node.py
    ros2 run <package> battle_node --ros-args -p raceline_csv:=/path/to/raceline.csv
"""

import copy
import csv
import math
import time
import os
from enum import IntEnum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, UInt8, UInt32
from visualization_msgs.msg import Marker

# === 感知常量 ===
DIR_DETECT_THRESHOLD = 2.5    # 方向检测距离阈值 (m)
OBS_DETECT_THRESHOLD = 5.0    # 障碍物检测距离阈值 (m)
MAX_SPEED_RATE = 2.0
THRESHOLD_obs = 0.5           # 障碍物边缘距离跳变阈值 (m)
THRESHOLD_TURN = 0.5          # 转弯检测阈值 (m)
START_ANGLE = -60
END_ANGLE = 60
MIN_OBS_SPEED = 1.0
DYNAMIC_INTENSITY_DIFF = 5    # 动态障碍物反射强度差异阈值
GAP_MIN_WIDTH = 18            # 可通行间隙最小角度宽度 (度)
OVERTAKE_GAP_WIDTH = 30       # 超车所需最小间隙宽度 (度)

CONTROL_MODE_HEURISTIC = 0
CONTROL_MODE_MPC_STEER = 1
CONTROL_MODE_MPC_FULL = 2
CONTROL_MODE_MPC_PP = 3


# ====================== 工具函数 ======================
def sigmoid(x: float, k: float = 1.0, x0: float = 0.0) -> float:
    """Sigmoid函数，用于融合权重的平滑映射
    公式：σ(x) = 1 / (1 + exp(-k * (x - x0)))
    """
    z = -k * (x - x0)
    if z > 500: return 0.0
    if z < -500: return 1.0
    return 1.0 / (1.0 + math.exp(z))


# ====================== 驾驶模式状态机 ======================
class DrivingMode(IntEnum):
    """驾驶模式枚举 —— 替代原有bool硬切换"""
    STRAIGHT = 0       # 无障直行
    STATIC_AVOID = 1   # 静态障碍避让
    FOLLOW = 2         # 动态跟随
    OVERTAKE = 3       # 超车


class ModeManager:
    """驾驶模式状态机 —— 带滞后确认窗口和速度平滑过渡
    
    设计原理：
    1. 候选模式需持续 confirm_duration_s 秒后才正式切换（防抖）
    2. 紧急安全场景（跟随/避障）缩短确认窗口至100ms
    3. 速度上限通过一阶低通滤波平滑过渡，避免硬跳变
    """
    def __init__(self, max_speed=9.0, follow_cap=1.0, overtake_cap=2.0,
                 confirm_s=0.3, smooth_alpha=0.15):
        self.current_mode = DrivingMode.STRAIGHT
        self.max_speed = max_speed
        self._candidate = DrivingMode.STRAIGHT
        self._timer = 0.0
        self._confirm_s = confirm_s
        self._alpha = smooth_alpha
        self._smoothed_bound = max_speed
        self._switch_count = 0
        self._speed_map = {
            DrivingMode.STRAIGHT: max_speed,
            DrivingMode.STATIC_AVOID: max_speed * 0.85,  # 静态避障：允许更高速通过
            DrivingMode.FOLLOW: follow_cap,
            DrivingMode.OVERTAKE: overtake_cap,
        }

    def update(self, dynamic_obs, follow, chaoche, has_obs, dt) -> DrivingMode:
        """更新模式状态机，返回当前模式"""
        # 场景分类
        if chaoche:       cand = DrivingMode.OVERTAKE
        elif follow or dynamic_obs: cand = DrivingMode.FOLLOW
        elif has_obs:     cand = DrivingMode.STATIC_AVOID
        else:             cand = DrivingMode.STRAIGHT

        # 带确认窗口的切换
        if cand != self.current_mode:
            if cand == self._candidate:
                self._timer += dt
            else:
                self._candidate = cand
                self._timer = dt
            threshold = min(0.1, self._confirm_s) if cand in (
                DrivingMode.FOLLOW, DrivingMode.STATIC_AVOID) else self._confirm_s
            if self._timer >= threshold:
                self.current_mode = cand
                self._timer = 0.0
                self._switch_count += 1
        else:
            self._timer = 0.0

        # 速度上限低通滤波平滑
        target = self._speed_map.get(self.current_mode, self.max_speed)
        self._smoothed_bound = (1 - self._alpha) * self._smoothed_bound + self._alpha * target
        return self.current_mode

    @property
    def speed_upper_bound(self): return self._smoothed_bound
    @property
    def mode_switch_count(self): return self._switch_count
    def get_mode_name(self):
        return {0: "直行加速", 1: "静态避让", 2: "动态跟随", 3: "超车"}.get(
            int(self.current_mode), "未知")


# ====================== 向量化MPC转向控制器 ======================
class MPCSteerSolver:
    """轻量化网格搜索MPC转向控制器（向量化版本）
    
    核心改进（相比原rollout_cost逐个循环）：
    1. NumPy批量并行计算所有候选转向角，消除Python循环
    2. 实现简化自行车动力学模型（含质心侧滑角β）
    3. 添加性能计时统计
    
    动力学模型：
        β = arctan(lr / (lf + lr) * tan(δ))
        x(k+1) = x(k) + v * cos(ψ + β) * dt
        y(k+1) = y(k) + v * sin(ψ + β) * dt
        ψ(k+1) = ψ(k) + v / lr * sin(β) * dt
    """
    def __init__(self, wheelbase=0.25, horizon=8, dt=0.1, candidates=11,
                 max_speed=9.0, min_speed=0.0, timeout_ms=8.0, use_dynamics=True):
        self.L = max(0.05, wheelbase)
        self.lr = wheelbase / 2.0
        self.lf = wheelbase / 2.0
        self.H = max(3, horizon)
        self.dt = max(0.02, dt)
        self.N = max(5, candidates)
        self.max_v = max_speed
        self.min_v = min_speed
        self.timeout_ms = timeout_ms
        self.use_dynamics = use_dynamics
        self.last_steer = 0.0
        # 代价权重
        self.w_x, self.w_y, self.w_yaw, self.w_v = 2.0, 3.0, 0.8, 0.3
        self.w_steer, self.w_dsteer = 0.05, 0.12
        # 预计算
        self._grid = np.linspace(-math.pi/4, math.pi/4, self.N)
        self._tan = np.tan(self._grid)
        # 统计
        self._count = 0; self._ok = 0; self._sum_ms = 0.0; self._max_ms = 0.0

    def solve(self, v0, ref, speed_bound):
        """向量化求解，返回 (success, steer, solve_ms)"""
        self._count += 1
        if ref is None:
            return False, self.last_steer, 0.0
        t0 = time.monotonic()
        N, H = self.N, self.H
        x = np.zeros(N); y = np.zeros(N); yaw = np.zeros(N)
        v = np.full(N, abs(v0)); cost = np.zeros(N)

        if self.use_dynamics:
            beta = np.arctan(self.lr / (self.lf + self.lr) * self._tan)
            for k in range(H):
                x += v * np.cos(yaw + beta) * self.dt
                y += v * np.sin(yaw + beta) * self.dt
                yaw += v / self.lr * np.sin(beta) * self.dt
                v = np.clip(v, self.min_v, speed_bound)
                ex = x - ref['x'][k]; ey = y - ref['y'][k]
                eyaw = (yaw - ref['yaw'][k] + math.pi) % (2*math.pi) - math.pi
                ev = v - ref['v'][k]
                cost += self.w_x*ex*ex + self.w_y*ey*ey + self.w_yaw*eyaw*eyaw + self.w_v*ev*ev
        else:
            for k in range(H):
                x += v * np.cos(yaw) * self.dt
                y += v * np.sin(yaw) * self.dt
                yaw += v / self.L * self._tan * self.dt
                v = np.clip(v, self.min_v, speed_bound)
                ex = x - ref['x'][k]; ey = y - ref['y'][k]
                eyaw = (yaw - ref['yaw'][k] + math.pi) % (2*math.pi) - math.pi
                ev = v - ref['v'][k]
                cost += self.w_x*ex*ex + self.w_y*ey*ey + self.w_yaw*eyaw*eyaw + self.w_v*ev*ev

        cost += self.w_steer * self._grid**2 * H
        cost += self.w_dsteer * (self._grid - self.last_steer)**2 * H
        ms = (time.monotonic() - t0) * 1000.0
        if ms > self.timeout_ms:
            return False, self.last_steer, ms
        best = int(np.argmin(cost))
        self.last_steer = float(self._grid[best])
        self._ok += 1; self._sum_ms += ms; self._max_ms = max(self._max_ms, ms)
        return True, self.last_steer, ms

    def stats(self):
        avg = self._sum_ms / max(1, self._ok)
        return f"avg={avg:.2f}ms max={self._max_ms:.2f}ms ok={self._ok}/{self._count}"


# ====================== 实验数据记录 ======================
@dataclass
class CycleRecord:
    """单个控制周期的数据记录（用于论文实验分析）"""
    timestamp: float = 0.0         # 运行时间 (s)
    lap_id: int = 0                # 圈次
    # 模式
    mode: int = 0                  # 模式枚举值
    mode_name: str = ""            # 模式名称
    # 控制指令
    steer: float = 0.0             # 转向角指令 (rad)
    speed: float = 0.0             # 速度指令 (m/s)
    actual_speed: float = 0.0      # 实际车速 (m/s)
    speed_upper_bound: float = 0.0 # 当前速度上限 (m/s)
    # MPC
    mpc_ok: bool = False           # MPC是否成功
    mpc_ms: float = 0.0            # MPC求解时间 (ms)
    # 融合
    blend_weight: float = 0.0      # 全局融合权重
    raceline_heading: float = 0.0  # Raceline目标航向 (rad)
    laser_heading: float = 0.0     # 激光反应式航向 (rad)
    fused_heading: float = 0.0     # 融合后航向 (rad)
    # 感知
    front_clearance: float = 0.0   # 前方净空距离 (m)
    num_obstacles: int = 0         # 障碍物数量
    # 位姿
    pos_x: float = 0.0             # X坐标 (m)
    pos_y: float = 0.0             # Y坐标 (m)
    heading: float = 0.0           # 航向角 (rad)
    # 跟踪误差
    lateral_error: float = 0.0     # 横向跟踪误差 (m)


class ExperimentLogger:
    """实验数据记录器
    
    使用方式：
    - 运行过程中每个控制周期调用 log()
    - Ctrl+C 退出时自动调用 save_all()
    - 数据保存到 ~/experiment_data/run_YYYYMMDD_HHMMSS/
    - 自动生成 summary.txt 摘要报告
    """
    def __init__(self, output_dir="~/experiment_data"):
        # 每次运行创建独立目录，避免覆盖
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(
            os.path.expanduser(output_dir), f"run_{timestamp_str}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self._records: List[CycleRecord] = []
        self._lap = 0
        self._t0 = time.time()

    def log(self, r: CycleRecord):
        r.timestamp = time.time() - self._t0
        r.lap_id = self._lap
        self._records.append(r)

    def finish_lap(self):
        if not self._records: return
        path = os.path.join(self.output_dir, f"lap_{self._lap:04d}.csv")
        self._save_csv(self._records, path)
        self._lap += 1; self._records = []

    def save_all(self) -> str:
        """保存所有数据 + 生成摘要报告，返回报告文本"""
        # 保存当前数据为完整运行CSV
        if not self._records:
            return "无数据记录"
        
        all_path = os.path.join(self.output_dir, "full_run.csv")
        self._save_csv(self._records, all_path)
        
        # 生成摘要报告
        report = self._generate_report(self._records)
        report_path = os.path.join(self.output_dir, "summary.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        self._records = []
        return report
    
    def _save_csv(self, records, path):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=asdict(records[0]).keys())
            w.writeheader()
            w.writerows([asdict(r) for r in records])
    
    def _generate_report(self, records: List[CycleRecord]) -> str:
        """生成本次运行的文字摘要报告"""
        if not records:
            return "无数据"
        
        n = len(records)
        duration = records[-1].timestamp - records[0].timestamp
        freq = n / max(duration, 0.001)
        
        speeds = [r.actual_speed for r in records]
        steers = [r.steer for r in records]
        mpc_times = [r.mpc_ms for r in records if r.mpc_ok]
        clearances = [r.front_clearance for r in records]
        weights = [r.blend_weight for r in records]
        
        # 转向角变化率（控制平滑性）
        steer_diffs = [abs(steers[i] - steers[i-1]) for i in range(1, len(steers))]
        steer_jerk = np.std(steer_diffs) if steer_diffs else 0
        
        # 模式统计
        from collections import Counter
        mode_counts = Counter(r.mode_name for r in records)
        mode_lines = [f"    {name}: {cnt}次 ({cnt/n*100:.1f}%)" 
                      for name, cnt in mode_counts.most_common()]
        
        lines = [
            "=" * 60,
            "  BattleVehicleNode 运行报告",
            "=" * 60,
            f"运行时长:       {duration:.1f} 秒",
            f"控制周期数:     {n} 次",
            f"控制频率:       {freq:.1f} Hz",
            f"数据保存路径:   {self.output_dir}",
            "",
            "--- 速度统计 ---",
            f"  平均速度:     {np.mean(speeds):.2f} m/s",
            f"  最大速度:     {np.max(speeds):.2f} m/s",
            f"  最小速度:     {np.min(speeds):.2f} m/s",
            f"  速度标准差:   {np.std(speeds):.3f} m/s",
            "",
            "--- MPC性能 ---",
            f"  成功率:       {len(mpc_times)}/{n} ({len(mpc_times)/n*100:.1f}%)",
        ]
        if mpc_times:
            lines += [
                f"  平均求解时间: {np.mean(mpc_times):.3f} ms",
                f"  最大求解时间: {np.max(mpc_times):.3f} ms",
                f"  P99求解时间:  {np.percentile(mpc_times, 99):.3f} ms",
            ]
        lines += [
            "",
            "--- 控制平滑性 ---",
            f"  转向角标准差:     {np.std(steers):.4f} rad ({np.degrees(np.std(steers)):.2f}°)",
            f"  转向变化率标准差: {steer_jerk:.4f} rad ({np.degrees(steer_jerk):.2f}°)",
            "",
            "--- 融合权重 ---",
            f"  平均权重:     {np.mean(weights):.3f} (1=纯Raceline, 0=纯激光)",
            f"  最小权重:     {np.min(weights):.3f}",
            "",
            "--- 前方净空 ---",
            f"  平均净空:     {np.mean(clearances):.2f} m",
            f"  最小净空:     {np.min(clearances):.2f} m",
            "",
            "--- 模式分布 ---",
        ] + mode_lines + [
            "",
            "=" * 60,
            f"完整数据: {self.output_dir}/full_run.csv",
            f"可视化:   python3 plot_run.py {self.output_dir}/full_run.csv",
            "=" * 60,
        ]
        return "\n".join(lines)


class BattleVehicleNode(Node):
    def __init__(self):
        super().__init__('battle_vehicle_node')
        
        # ====================== Raceline 参数 ======================
        self.declare_parameter('raceline_csv', '/home/jetson/Raceline-Optimization/inputs/tracks/Spielberg_map.csv')
        self.raceline_array = None
        self.raceline_path_msg = None
        self.raceline_total_length = 0.0
        self.raceline_sample_s = None
        self.raceline_sample_x = None
        self.raceline_sample_y = None
        self.raceline_sample_yaw = None
        self.raceline_sample_v = None
        self.last_raceline_index = None
        
        raceline_path = self.get_parameter('raceline_csv').value
        if raceline_path and os.path.exists(raceline_path):
            self._load_raceline(raceline_path)
        else:
            self.get_logger().warn(f"⚠️ 未提供 raceline 或文件不存在: {raceline_path}，将退化为纯反应式")
            
        # Raceline 加载完成后添加：
        self.raceline_pub = self.create_publisher(Path, '/raceline', 1)
        self.raceline_timer = self.create_timer(0.5, self.publish_raceline_timer)
        
        # ---- 话题参数: scan前端输入、最终控制输出、可视化调试与里程计输入 ----
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('marker_topic', '/arrow_marker_02')
        self.declare_parameter('debug_scan_topic', '/front_scan_02')
        self.declare_parameter('odom_topic', '/odom')
        
        # ---- MPC参数 ----
        self.declare_parameter('mpc_enable', True)
        self.declare_parameter('mpc_mode', 'steer_only')
        self.declare_parameter('mpc_timeout_ms', 8.0)
        self.declare_parameter('wheelbase', 0.25)
        self.declare_parameter('horizon', 8)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('mpc_steer_candidates', 11)
        self.declare_parameter('mpc_accel_candidates', 7)
        self.declare_parameter('mpc_max_speed', 9.0)
        self.declare_parameter('mpc_min_speed', 0.0)
        self.declare_parameter('mpc_max_accel', 2.0)
        self.declare_parameter('mpc_max_decel', -2.0)
        
        # Phase 2 速度边界
        self.declare_parameter('follow_speed_cap', 1.5)    # 跟随速度上限：从1.0提高到1.5
        self.declare_parameter('chaoche_speed_cap', 4.5)    # 超车速度上限：从2.0大幅提高到4.5
        self.declare_parameter('final_speed_cap', 12.0)     # 全局速度上限：从10.0提高到12.0
        
        # Phase 3诊断
        self.declare_parameter('diag_print_hz', 2.0)
        self.declare_parameter('publish_diag_topics', True)
        
        # safety 接口
        self.declare_parameter('safety_pre_cmd_topic', '/battle_fast2/pre_safety_drive')
        self.declare_parameter('safety_status_topic', '/battle_fast2/safety_status')
        
        # Pure Pursuit
        self.declare_parameter('pure_pursuit_enable', True)
        self.declare_parameter('pp_lookahead', 1.2)
        
        # 读取参数
        scan_topic = self.get_parameter('scan_topic').value
        drive_topic = self.get_parameter('drive_topic').value
        marker_topic = self.get_parameter('marker_topic').value
        debug_scan_topic = self.get_parameter('debug_scan_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        
        self.mpc_enable = bool(self.get_parameter('mpc_enable').value)
        self.mpc_mode = str(self.get_parameter('mpc_mode').value)
        self.mpc_timeout_ms = float(self.get_parameter('mpc_timeout_ms').value)
        self.wheelbase = max(0.05, float(self.get_parameter('wheelbase').value))
        self.mpc_horizon = max(3, int(self.get_parameter('horizon').value))
        self.mpc_dt = max(0.02, float(self.get_parameter('dt').value))
        self.mpc_steer_candidates = max(5, int(self.get_parameter('mpc_steer_candidates').value))
        self.mpc_accel_candidates = max(3, int(self.get_parameter('mpc_accel_candidates').value))
        self.mpc_max_speed = float(self.get_parameter('mpc_max_speed').value)
        self.mpc_min_speed = float(self.get_parameter('mpc_min_speed').value)
        self.mpc_max_accel = float(self.get_parameter('mpc_max_accel').value)
        self.mpc_max_decel = float(self.get_parameter('mpc_max_decel').value)
        
        self.follow_speed_cap = float(self.get_parameter('follow_speed_cap').value)
        self.chaoche_speed_cap = float(self.get_parameter('chaoche_speed_cap').value)
        self.final_speed_cap = float(self.get_parameter('final_speed_cap').value)
        
        self.diag_print_hz = max(0.2, float(self.get_parameter('diag_print_hz').value))
        self.publish_diag_topics = bool(self.get_parameter('publish_diag_topics').value)
        
        safety_pre_cmd_topic = self.get_parameter('safety_pre_cmd_topic').value
        safety_status_topic = self.get_parameter('safety_status_topic').value
        
        self.pure_pursuit_enable = bool(self.get_parameter('pure_pursuit_enable').value)
        self.pp_lookahead = float(self.get_parameter('pp_lookahead').value)
        
        self.get_logger().info(
            f"Battle Node Started. scan={scan_topic}, drive={drive_topic}, odom={odom_topic}, "
            f"mpc_enable={self.mpc_enable}, mpc_mode={self.mpc_mode}, "
            f"pure_pursuit_enable={self.pure_pursuit_enable} (PP控制速度), pp_lookahead={self.pp_lookahead:.2f}m, "
            f"raceline_loaded={'True' if self.raceline_array is not None else 'False'}"
        )
        
        # 原有反应式状态
        self.last_angle = 0.0
        self.last_max_dir_index = 0
        self.GO_STARIGHT = 0
        self.TRANSITION = 0
        self.last_in_normol = False
        self.last_in_straight = False
        self.speed_rate = 1.0
        self.straight_cnt = 0
        self.Follow = False
        self.turn_rate = 1.0
        self.P = 1.1
        self.D = 0.2
        self.dynamic_obs = False
        self.chaoche = False
        
        # Odom 与控制状态
        self.have_odom = False
        self.odom_speed = 0.0
        self.odom_yaw = 0.0
        self.odom_pose = None
        
        self.last_mpc_steer = 0.0
        self.last_mpc_speed = 0.0
        self.last_control_mode = CONTROL_MODE_HEURISTIC
        self.current_speed_upper_bound = self.mpc_max_speed
        self.last_mpc_reason = 'init'
        
        # 统计
        self.control_cycle_count = 0
        self.mpc_attempt_count = 0
        self.mpc_success_count = 0
        self.mode_switch_count = 0
        self.last_solve_time_ms = 0.0
        self.max_solve_time_ms = 0.0
        self.sum_solve_time_ms = 0.0
        
        self.have_safety_status = False
        self.safety_status = False
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.middle_line_callback, qos_profile)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 20)
        self.safety_status_sub = self.create_subscription(Bool, safety_status_topic, self.safety_status_callback, 10)
        
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.scan_pub = self.create_publisher(LaserScan, debug_scan_topic, 10)
        self.marker_pub = self.create_publisher(Marker, marker_topic, 1)
        
        self.heuristic_pub = self.create_publisher(AckermannDriveStamped, 'battle_fast2/drive_heuristic', 10)
        self.mpc_pub = self.create_publisher(AckermannDriveStamped, 'battle_fast2/drive_mpc', 10)
        self.mode_pub = self.create_publisher(UInt8, 'battle_fast2/control_mode', 10)
        self.mpc_ok_pub = self.create_publisher(Bool, 'battle_fast2/mpc_ok', 10)
        self.solve_time_pub = self.create_publisher(Float32, 'battle_fast2/mpc_solve_time_ms', 10)
        self.mode_switch_pub = self.create_publisher(UInt32, 'battle_fast2/mode_switch_count', 10)
        self.current_speed_pub = self.create_publisher(Float32, 'battle_fast2/current_speed_mps', 10)
        self.pre_safety_pub = self.create_publisher(AckermannDriveStamped, safety_pre_cmd_topic, 10)
        
        self.diag_timer = self.create_timer(1.0 / self.diag_print_hz, self.diagnostic_timer_callback)
        
        # ====================== 新增模块初始化 ======================
        # 模式状态机（替代原有bool硬切换，带平滑过渡）
        self.mode_mgr = ModeManager(
            max_speed=self.mpc_max_speed,
            follow_cap=self.follow_speed_cap,
            overtake_cap=self.chaoche_speed_cap,
        )
        # 向量化MPC求解器（含动力学模型）
        self.mpc_solver = MPCSteerSolver(
            wheelbase=self.wheelbase,
            horizon=self.mpc_horizon,
            dt=self.mpc_dt,
            candidates=self.mpc_steer_candidates,
            max_speed=self.mpc_max_speed,
            min_speed=self.mpc_min_speed,
            timeout_ms=self.mpc_timeout_ms,
            use_dynamics=True,
        )
        # 实验数据记录器
        self.exp_logger = ExperimentLogger()
        
        self.get_logger().info(
            f"节点启动完成 | 动力学模型={'开启' if self.mpc_solver.use_dynamics else '关闭'}"
            f" | 模式管理={'启用'} | 实验记录={'启用'}"
        )

    def odom_callback(self, msg: Odometry):
        self.have_odom = True
        self.odom_speed = float(msg.twist.twist.linear.x)
        
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.odom_pose = PoseStamped()
        self.odom_pose.header = msg.header
        self.odom_pose.pose = copy.deepcopy(msg.pose.pose)

    def safety_status_callback(self, msg: Bool):
        self.have_safety_status = True
        self.safety_status = bool(msg.data)

    def diagnostic_timer_callback(self):
        self.get_logger().info(
            f"v={self.odom_speed:.2f}m/s | 模式={self.mode_mgr.get_mode_name()} "
            f"| MPC: {self.mpc_solver.stats()} "
            f"| 模式切换={self.mode_mgr.mode_switch_count}次"
        )
        if self.publish_diag_topics:
            solve_msg = Float32()
            solve_msg.data = float(self.last_solve_time_ms)
            self.solve_time_pub.publish(solve_msg)
            
            switch_msg = UInt32()
            switch_msg.data = int(self.mode_mgr.mode_switch_count)
            self.mode_switch_pub.publish(switch_msg)
            
            speed_msg = Float32()
            speed_msg.data = float(self.odom_speed)
            self.current_speed_pub.publish(speed_msg)

    def publish_arrow_marker(self, max_dir_index, frame_id='laser'):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'direction_arrow'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        p0 = Point(x=0.0, y=0.0, z=0.0)
        angle_rad = math.radians(max_dir_index)
        p1 = Point(x=math.sin(angle_rad), y=math.cos(angle_rad), z=0.0)
        marker.points = [p0, p1]
        
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.1
        marker.color.g = 1.0
        marker.color.b = 0.1
        marker.lifetime = Duration(sec=0, nanosec=100000000)
        
        self.marker_pub.publish(marker)

    # ====================== 以下所有 frontend 函数保持完全不变 ======================
    def get_dis(self, data, angle, deg=True, return_inten=True):
        if deg:
            angle = np.deg2rad(angle)
        dis = 0
        intensities = None
        temp = int((angle - data.angle_min) / data.angle_increment)
        start_idx = max(0, temp - 2)
        end_idx = min(len(data.ranges), temp + 2)
        
        data_tmp = data.ranges[start_idx:end_idx]
        inten_tmp = data.intensities[start_idx:end_idx] if len(data.intensities) > 0 else [0] * len(data_tmp)
        
        data_tmp = np.sort(np.array(data_tmp))
        inten_tmp = np.sort(np.array(inten_tmp))
        
        if len(data_tmp) > 2:
            dis = data_tmp[2]
            intensities = inten_tmp[2]
        elif len(data_tmp) > 0:
            dis = data_tmp[0]
            intensities = inten_tmp[0]
            
        if return_inten:
            return dis, intensities
        return dis

    def get_range(self, data, start_angle, end_engle, return_inten=False):
        all_dis = []
        all_inten = []
        for angle in range(start_angle, end_engle):
            tmp = self.get_dis(data, angle, return_inten=return_inten)
            all_dis.append(tmp[0])
            all_inten.append(tmp[1])
        if return_inten:
            return all_dis, all_inten
        return all_dis

    def fill_zeros_with_neighbors(self, data):
        result = list(data)
        n = len(result)
        for i in range(n):
            if result[i] == 0:
                left = next((result[j] for j in range(i - 1, -1, -1) if result[j] != 0), None)
                if left is not None:
                    result[i] = left
                    continue
                right = next((result[j] for j in range(i + 1, n) if result[j] != 0), None)
                if right is not None:
                    result[i] = right
                    continue
                result[i] = 0
        return result

    def filter_obstacles_by_variance(self, Left_obs_orig, dis_90, variance_threshold=1.0):
        Left_obs = []
        if len(Left_obs_orig) > 0:
            for i in range(0, int(len(Left_obs_orig) / 2), 1):
                idx_start = int(Left_obs_orig[2 * i])
                idx_end = int(Left_obs_orig[2 * i + 1])
                obstacle_range = dis_90[idx_start: idx_end]
                dis_obs_var = np.var(obstacle_range)
                print('in filter_obstacles_by_variance,方差：', dis_obs_var)
                Left_obs.append(idx_start)
                Left_obs.append(idx_end)
        return Left_obs

    def filter_anomalous_values(self, data, max_distance=4, angle_range=2):
        data = np.array(data)
        for i in range(1, len(data) - 1):
            if data[i] != max_distance:
                if all(data[j] == max_distance for j in range(i - angle_range, i + angle_range + 1) if 0 <= j < len(data)):
                    data[i] = (data[i - 1] + data[i + 1]) / 2
        return data.tolist()

    def filter_small_obstacles(self, Left_obs, min_obstacle_size=2):
        for i in range(int(len(Left_obs) / 2)):
            if abs(Left_obs[2 * i] - Left_obs[2 * i + 1]) <= min_obstacle_size:
                Left_obs[2 * i] = -1
                Left_obs[2 * i + 1] = -1
        Left_obs_temp = Left_obs
        Left_obs = [x for x in Left_obs_temp if x != -1]
        return Left_obs

    def pub_scan(self, dis_90, header):
        scan_msg = LaserScan()
        scan_msg.header = header
        scan_msg.angle_min = np.pi / 2
        scan_msg.angle_max = -np.pi / 2
        scan_msg.angle_increment = -np.pi / 180
        scan_msg.ranges = [float(x) for x in dis_90]
        scan_msg.intensities = []
        scan_msg.range_max = 100.0
        self.scan_pub.publish(scan_msg)

    def DynamicObastcle(self, dis_list, inten_list, max_dir_num, obs):
        if len(max_dir_num) < 3:
            return False
        max_range = [max_dir_num[0], max_dir_num[-1]]
        obs_range = [max_dir_num[1], max_dir_num[2]]
        range_list = inten_list[max_range[0]:max_range[1]]
        obs_intensity = inten_list[obs_range[0]:obs_range[1]]
        if len(range_list) == 0 or len(obs_intensity) == 0:
            return False
        average_obs_intensity = np.mean(obs_intensity)
        average_intensity = np.mean(range_list)
        abs_tmp = abs(average_obs_intensity - average_intensity)
        if abs_tmp > 5:
            print('detect dynamic obs abs_tmpabs_tmpabs_tmpabs_tmp!!!!', abs_tmp)
            return True
        return False

    def frontend_scan_process(self, data):
        # （以下所有代码与你原来完全一致，没有任何改动）
        self.dynamic_obs = False
        self.chaoche = False
        self.Follow = False
        self.D = 0.2
        print('###########################################################')
        
        dis_90, inten_90 = self.get_range(data, -89, 91, True)
        dis_90 = dis_90[::-1]
        inten_90 = inten_90[::-1]
        dis_obs_90 = copy.deepcopy(dis_90)
        lenth_dis = len(dis_90)
        left = 0
        right = 0
        Left_obs_orig = []
        Left_obs = []
        max_dis_num = []
        max_dir_num = []
        max_dis = 0
        max_dir_range = 0
        max_dis_index = 0
        max_dir_index = 0
        
        self.pub_scan(dis_90, data.header)
        
        dis_90 = self.fill_zeros_with_neighbors(dis_90)
        inten_90 = self.fill_zeros_with_neighbors(inten_90)
        dis_obs_90 = self.fill_zeros_with_neighbors(dis_obs_90)
        dis_90_copy = tuple(dis_90)
        
        for i in range(0, lenth_dis, 1):
            if dis_90[i] > max_dis and i > 20 and i < 160:
                max_dis = dis_90[i]
                max_dis_index = i
            if dis_90[i] > DIR_DETECT_THRESHOLD:
                dis_90[i] = DIR_DETECT_THRESHOLD
            if dis_obs_90[i] > OBS_DETECT_THRESHOLD:
                dis_obs_90[i] = OBS_DETECT_THRESHOLD
                
        dis_90 = self.filter_anomalous_values(dis_90, max_distance=DIR_DETECT_THRESHOLD, angle_range=2)
        dis_obs_90 = self.filter_anomalous_values(dis_obs_90, max_distance=OBS_DETECT_THRESHOLD, angle_range=2)
        
        if max_dis_index < 89:
            left = 1
        else:
            right = 1
            
        for i in range(0, lenth_dis - 2, 1):
            if dis_obs_90[i] - dis_obs_90[i + 1] > THRESHOLD_obs and len(Left_obs_orig) % 2 == 0:
                Left_obs_orig.append(i + 1)
            elif dis_obs_90[i + 1] - dis_obs_90[i] > THRESHOLD_obs and len(Left_obs_orig) % 2 == 1:
                Left_obs_orig.append(i)
                
        if len(Left_obs_orig) % 2 == 1:
            print('error!!!!!!!!!!!!')
            Left_obs_orig.pop()
            
        Left_obs = self.filter_small_obstacles(Left_obs_orig, min_obstacle_size=2)
        Left_obs = self.filter_obstacles_by_variance(Left_obs, dis_obs_90, variance_threshold=1.0)
        
        if len(Left_obs) > 0:
            for i in range(0, int(len(Left_obs) / 2), 1):
                idx_mid = int((Left_obs[2 * i] + Left_obs[2 * i + 1]) / 2)
                obs_middle = dis_obs_90[idx_mid]
                start_expand = int(max(Left_obs[2 * i] - min((Left_obs[2 * i + 1] - Left_obs[2 * i]) / 2 * (4 - obs_middle), 10), 0))
                end_expand = int(min(Left_obs[2 * i + 1] + min((Left_obs[2 * i + 1] - Left_obs[2 * i]) / 2 * (4 - obs_middle), 10), lenth_dis - 1))
                for j in range(start_expand, end_expand, 1):
                    dis_obs_90[j] = obs_middle
                Left_obs[2 * i] = start_expand
                Left_obs[2 * i + 1] = end_expand
            print('有障碍物，障碍物是', Left_obs)
        else:
            print('没有障碍物')
            
        if len(Left_obs) > 0:
            for i in range(0, int(len(Left_obs) / 2) + 1, 1):
                if i == 0:
                    for j in range(Left_obs[0] - 1, 0, -1):
                        if dis_obs_90[j] <= dis_obs_90[Left_obs[0] + 1]:
                            max_dis_num.append(j)
                            max_dis_num.append(Left_obs[0] + 1)
                            break
                    if len(max_dis_num) == 0:
                        for j in range(0, Left_obs[0] - 1, 1):
                            if dis_obs_90[j] >= dis_obs_90[Left_obs[0] + 1]:
                                max_dis_num.append(j)
                                max_dis_num.append(Left_obs[0] + 1)
                                break
                elif i < int(len(Left_obs) / 2):
                    max_dis_num.append(Left_obs[2 * i - 1])
                    max_dis_num.append(Left_obs[2 * i])
                elif i == int(len(Left_obs) / 2):
                    for j in range(Left_obs[2 * i - 1] + 1, lenth_dis - 1, 1):
                        if dis_obs_90[j] <= dis_obs_90[Left_obs[2 * i - 1] - 1]:
                            max_dis_num.append(Left_obs[2 * i - 1] - 1)
                            max_dis_num.append(j)
                            break
                            
            max_dis_val = 0
            max_dis_index_temp = max_dis_index
            for i in range(0, int(len(max_dis_num) / 2), 1):
                if max_dis_val < max_dis_num[2 * i + 1] - max_dis_num[2 * i]:
                    max_dis_val = max_dis_num[2 * i + 1] - max_dis_num[2 * i]
                    max_dis_index = (max_dis_num[2 * i + 1] + max_dis_num[2 * i]) / 2
                    
            if left == 1 and max_dis_index < 90 and dis_obs_90[0] < dis_obs_90[lenth_dis - 1] - 1:
                max_dis_index = max_dis_index + 5 * abs(dis_obs_90[lenth_dis - 1] - dis_obs_90[0])
            elif right == 1 and max_dis_index > 90 and dis_obs_90[0] - 1 > dis_obs_90[lenth_dis - 1]:
                max_dis_index = max_dis_index - 5 * abs(dis_obs_90[lenth_dis - 1] - dis_obs_90[0])
                
            print('max_dis_index_temp', max_dis_index_temp)
            
            if len(Left_obs) == 2:
                if max_dis_index_temp >= 89:
                    print('turn right')
                    if Left_obs[0] >= 89:
                        max_dis_index = int((89 + Left_obs[0]) / 2)
                    elif Left_obs[1] <= 89:
                        max_dis_index = max_dis_index_temp
                    else:
                        max_dis_index = max_dis_index_temp
                else:
                    if Left_obs[0] >= 89:
                        print('1')
                        max_dis_index = max_dis_index_temp
                    elif Left_obs[1] <= 89:
                        print('2')
                        max_dis_index = int((89 + Left_obs[1]) / 2)
                    else:
                        print('3')
                        max_dis_index = max_dis_index_temp
                        
            if len(Left_obs) == 4:
                middle_temp = int((Left_obs[1] + Left_obs[2]) / 2)
                if max_dis_index_temp >= 89:
                    if Left_obs[0] >= 89:
                        max_dis_index = int((89 + Left_obs[0]) / 2)
                    elif Left_obs[1] <= 89 and middle_temp > 89:
                        max_dis_index = int((Left_obs[1] + max_dis_index_temp) / 2)
                    elif middle_temp <= 89 and Left_obs[2] > 89:
                        max_dis_index = middle_temp
                    else:
                        max_dis_index = max_dis_index_temp
                else:
                    print('turn left')
                    if Left_obs[0] > 89:
                        max_dis_index = max_dis_index_temp
                    elif Left_obs[0] <= 89 and middle_temp > 89:
                        max_dis_index = middle_temp
                    elif middle_temp <= 89 and Left_obs[3] > 89:
                        max_dis_index = middle_temp
                    else:
                        max_dis_index = int((Left_obs[3] + 89) / 2)
                        
        for i in range(0, lenth_dis - 2, 1):
            if dis_90[i] < DIR_DETECT_THRESHOLD and dis_90[i + 1] == DIR_DETECT_THRESHOLD and len(max_dir_num) % 2 == 0:
                max_dir_num.append(i + 1)
            elif dis_90[i] == DIR_DETECT_THRESHOLD and dis_90[i + 1] < DIR_DETECT_THRESHOLD and len(max_dir_num) % 2 == 1:
                max_dir_num.append(i)
                
        if len(max_dir_num) % 2 == 1 and len(max_dir_num) != 1:
            self.get_logger().error('出现单个不封闭区域，请检查障碍物检测逻辑')
            
        if len(max_dir_num) == 1:
            if max_dir_num[0] < 90:
                max_dir_index = int((max_dir_num[0]) / 2)
                print('\033[32m转弯阶段1左转，最大距离朝向: %s\033[0m' % (max_dir_index - lenth_dis / 2))
            elif max_dir_num[0] > 90:
                max_dir_index = int((max_dir_num[0] + lenth_dis - 2) / 2)
                print('\033[32m转弯阶段1右转，最大距离朝向: %s\033[0m' % (max_dir_index - lenth_dis / 2))
            self.GO_STARIGHT = 0
            
        if len(max_dir_num) == 2:
            print('找到 %d 个最大距离区域，区域大小 %d' % ((int(len(max_dir_num) / 2)), max_dir_num[1] - max_dir_num[0]))
            max_dir_index = int((max_dir_num[0] + max_dir_num[1]) / 2)
            max_dir_range = max_dir_num[1] - max_dir_num[0]
            
        if len(max_dir_num) > 2:
            print('有多个最大距离区域，障碍物个数为:', len(Left_obs) / 2)
            if len(Left_obs) > 0:
                self.dynamic_obs = self.DynamicObastcle(dis_list=dis_90, inten_list=inten_90, max_dir_num=max_dir_num, obs=Left_obs)
            cand_space = []
            cand_dirs = []
            for i in range(0, int(len(max_dir_num) / 2), 1):
                cand_space.append(max_dir_num[2 * (i) + 1] - max_dir_num[2 * (i)])
                cand_dirs.append((max_dir_num[2 * (i) + 1] + max_dir_num[2 * (i)]) / 2)
                
            cand_dir_id = np.where(np.array(cand_space) > 18)[0]
            if len(cand_dir_id) != 0:
                selected_dirs = np.array(cand_dirs)[cand_dir_id].tolist()
                max_dir_idx = np.argmin(selected_dirs)
                selected_ranges = np.array(cand_space)[cand_dir_id].tolist()
                max_dir_index = selected_dirs[max_dir_idx]
                max_dir_range = selected_ranges[max_dir_idx]
                cand_dir_chaoche_idx = np.where(np.array(cand_space) > 30)[0]
                
                if self.dynamic_obs:
                    if cand_dir_chaoche_idx.size:
                        self.chaoche = True
                    else:
                        self.Follow = True
                else:
                    cand_dir_id = np.argmax(np.array(cand_space))
                    max_dir_index = cand_dirs[cand_dir_id]
                    max_dir_range = cand_space[cand_dir_id]
                    
        if self.dynamic_obs:
            if len(max_dir_num) >= 3:
                max_dir_index = int((max_dir_num[1] + max_dir_num[2]) / 2)
                max_dir_range = max_dir_num[-1] - max_dir_num[0]
            self.Follow = True
            if max_dir_index < 90:
                max_dir_index -= 2
            else:
                max_dir_index += 2
                
        print(max_dir_num, self.P)
        
        if max_dir_index >= 75 and max_dir_index <= 105:
            mean_straight = np.mean(dis_90_copy[80:100])
            self.GO_STARIGHT = 1
            self.TRANSITION = 0
            if self.last_in_straight and max_dir_range > 20:
                self.speed_rate *= 8.05
                # 直道速度上限：根据前方净空距离分级（提高上限）
                if mean_straight > 11 and len(Left_obs) == 0:
                    limit_rate = 10.0   # 超远距离直道：全力加速
                elif mean_straight > 8 and len(Left_obs) == 0:
                    limit_rate = 7.5    # 远距离直道
                elif mean_straight > 7 and len(Left_obs) == 0:
                    limit_rate = 5.0    # 中距离直道
                elif mean_straight > 5 and len(Left_obs) == 0:
                    limit_rate = 4.0    # 短距离直道
                else:
                    limit_rate = 3.0
                if self.speed_rate > limit_rate:
                    self.speed_rate = limit_rate
            else:
                self.speed_rate = 1.3   # 直道起步增益：从1.1提高到1.3
            self.last_in_straight = True
        elif max_dir_index < 75 and max_dir_index > 0:
            # 弯道速度：根据偏转角度分级，不再一刀切为1.0
            turn_offset = abs(max_dir_index - 90)  # 偏转角度（度）
            self.P = 1.5
            if turn_offset < 25:        # 小弯（偏转<25°）
                self.speed_rate = 1.8
                self.turn_rate = 0.9
            elif turn_offset < 45:      # 中弯
                self.speed_rate = 1.4
                self.turn_rate = 0.85
            else:                       # 大弯
                self.speed_rate = 1.1
                self.turn_rate = 0.8
            self.last_in_straight = False
        elif max_dir_index > 105:
            turn_offset = abs(max_dir_index - 90)
            self.P = 1.5
            if turn_offset < 25:
                self.speed_rate = 1.8
                self.turn_rate = 0.9
            elif turn_offset < 45:
                self.speed_rate = 1.4
                self.turn_rate = 0.85
            else:
                self.speed_rate = 1.1
                self.turn_rate = 0.8
            self.last_in_straight = False
            
        normol = 1
        if len(max_dir_num) == 0:
            if self.GO_STARIGHT == 1 or self.TRANSITION == 1:
                for i in range(0, lenth_dis - 2, 1):
                    if dis_90[i + 1] - dis_90[i] > THRESHOLD_TURN:
                        max_dir_index = int((i + 1 + len(dis_90) / 2) / 2)
                        self.get_logger().warn('进入过渡路段，前方左转，最大距离朝向: %f' % (max_dir_index - len(dis_90) / 2))
                        self.P = 0.8
                        normol = 0
                    elif dis_90[i] - dis_90[i + 1] > THRESHOLD_TURN:
                        max_dir_index = int((i + len(dis_90) / 2) / 2)
                        self.get_logger().warn('进入过渡路段，前方右转，最大距离朝向: %f' % (max_dir_index - len(dis_90) / 2))
                        self.P = 0.8
                        normol = 0
            if normol == 1:
                max_dir_index = self.last_max_dir_index
                print('\033[38;5;208m隐藏款，无法判断方向，保持上一次动作并减速，最大距离朝向: %f\033[0m' % (max_dir_index - len(dis_90) / 2))
                if self.last_in_normol:
                    self.speed_rate *= 0.9
                    self.turn_rate *= 1.2
                    if self.speed_rate < 0.5:
                        self.speed_rate = 0.5
                    if self.turn_rate > 2.5:
                        self.turn_rate = 2.5
                else:
                    self.speed_rate = 0.9
                    self.turn_rate = 1.2
                self.last_in_normol = True
            else:
                self.speed_rate = 1.0
                self.turn_rate = 1.0
                self.last_in_normol = False
            self.TRANSITION = 1
            self.GO_STARIGHT = 0
            
        dis_90[0] = dis_90[0] + 0.00001
        dis_90[lenth_dis - 1] = dis_90[lenth_dis - 1] + 0.00001
        
        return {
            'dis_90': dis_90,
            'lenth_dis': lenth_dis,
            'max_dis': max_dis,
            'max_dir_index': max_dir_index,
            'max_dir_num': max_dir_num,
            'left_obs': Left_obs,
            'header': data.header,
        }

    def heuristic_backend_control(self, front_state):
        # （完全不变）
        dis_90 = front_state['dis_90']
        lenth_dis = front_state['lenth_dis']
        max_dis = front_state['max_dis']
        max_dir_index = front_state['max_dir_index']
        print('视野中最大距离是', max_dis)
        angle = 0.0
        
        if max_dir_index != 0:
            term1 = -max(math.exp(-max_dis / DIR_DETECT_THRESHOLD), 0.7) * (max_dir_index - 90) / 360 * math.pi
            term2 = (dis_90[0] - dis_90[lenth_dis - 1]) / (dis_90[0] + dis_90[lenth_dis - 1])
            print(f'term1:{term1}, term2:{term2}')
            if dis_90[0] / dis_90[lenth_dis - 1] > 3 or dis_90[lenth_dis - 1] / dis_90[0] > 3:
                self.D = 0.5
                print('边界！！！！！！！！！！！！！！！！')
                angle = 1.0 * term1 + 0.05 * term2
            else:
                angle = 1.0 * term1 + 0.02 * term2
        else:
            if dis_90[0] / dis_90[lenth_dis - 1] > 3 or dis_90[lenth_dis - 1] / dis_90[0] > 3:
                angle = -max(math.exp(-max_dis / DIR_DETECT_THRESHOLD), 0.7) * (max_dir_index - 90) / 360 * math.pi + 0.1 * (dis_90[0] - dis_90[lenth_dis - 1]) / (dis_90[0] + dis_90[lenth_dis - 1])
            else:
                angle = -max(math.exp(-max_dis / DIR_DETECT_THRESHOLD), 0.7) * (max_dir_index - 90) / 360 * math.pi + 0.05 * (dis_90[0] - dis_90[lenth_dis - 1]) / (dis_90[0] + dis_90[lenth_dis - 1])
                
        steering_angle = self.P * angle + self.D * (angle - self.last_angle)
        self.last_angle = angle
        speed = 1.8 * (0.3 * math.exp(-np.clip(abs(angle), 0, 0.5)) + 0.7)
        steering_angle = self.turn_rate * steering_angle
        steering_angle = np.clip(steering_angle, -math.pi / 4, math.pi / 4)
        
        return {
            'angle': angle,
            'steering': float(steering_angle),
            'speed': float(self.speed_rate * speed),
            'base_speed': float(speed),
        }

    # ====================== 【核心修改】Raceline + 反应式融合 ======================
    def _wrap_angle(self, angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _blend_angle(self, angle_a, angle_b, weight_b):
        weight_b = float(np.clip(weight_b, 0.0, 1.0))
        return self._wrap_angle(angle_a + weight_b * self._wrap_angle(angle_b - angle_a))

    def _pose_yaw(self, pose):
        if pose is None:
            return 0.0
        qx = pose.pose.orientation.x
        qy = pose.pose.orientation.y
        qz = pose.pose.orientation.z
        qw = pose.pose.orientation.w
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def _world_to_body(self, dx, dy, yaw):
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        body_x = cos_yaw * dx + sin_yaw * dy
        body_y = -sin_yaw * dx + cos_yaw * dy
        return body_x, body_y

    def _compute_raceline_blend_weight(self, front_state):
        """双层自适应融合权重计算（论文核心公式）
        
        第一层 - 场景语义基准权重：w_base = f(mode)
        第二层 - 前方净空sigmoid动态修正：
            w_clearance = σ(k * (d_front - d_th))，映射到[0.35, 0.95]
        最终：w = min(w_base, w_clearance)
        """
        # 第一层：基于模式状态机的基准权重
        base_map = {
            DrivingMode.STRAIGHT: 0.92,
            DrivingMode.STATIC_AVOID: 0.75,
            DrivingMode.FOLLOW: 0.50,
            DrivingMode.OVERTAKE: 0.65,
        }
        w_base = base_map.get(self.mode_mgr.current_mode, 0.85)
        
        # 第二层：前方净空距离的sigmoid动态修正
        front_window = front_state['dis_90'][80:101]
        d_front = float(np.min(front_window)) if len(front_window) > 0 else DIR_DETECT_THRESHOLD
        # sigmoid: d_front越小 → w_clearance越小 → 越偏向局部避障
        k, d_th = 3.0, 1.5  # sigmoid陡峭度和中心距离
        raw = sigmoid(d_front, k=k, x0=d_th)
        w_clearance = 0.35 + 0.60 * raw  # 映射到 [0.35, 0.95]
        
        return float(np.clip(min(w_base, w_clearance), 0.35, 0.95))

    def _load_raceline(self, csv_path):
        """加载 Raceline 并计算缺失的 yaw 和速度"""
        df = pd.read_csv(csv_path, comment='#', header=None)
        raw_array = df.to_numpy(dtype=np.float32)
        self.get_logger().info(f"原始 CSV shape={raw_array.shape}, 列数={raw_array.shape[1]}")
        
        # 【修复】检查列数并补充缺失数据
        if raw_array.shape[1] == 4:
            # 格式: [x, y, w_right, w_left]
            x_vals = raw_array[:, 0]
            y_vals = raw_array[:, 1]
            
            # 计算 yaw（相邻点的切线方向）
            dx = np.diff(x_vals, append=x_vals[0])
            dy = np.diff(y_vals, append=y_vals[0])
            yaw_vals = np.arctan2(dy, dx)
            yaw_unwrapped = np.unwrap(yaw_vals)
            
            # 计算弧长
            distances = np.sqrt(dx**2 + dy**2)
            s_vals = np.concatenate([[0.0], np.cumsum(distances[:-1])])
            
            # 简单速度计算：曲率越小速度越快
            closing_yaw = yaw_unwrapped[-1] + self._wrap_angle(yaw_unwrapped[0] - yaw_unwrapped[-1])
            curvature = np.abs(np.diff(yaw_unwrapped, append=closing_yaw)) / (distances + 1e-6)
            v_opt = 3.0 / (1.0 + 0.5 * curvature) # 基础速度 3.0 m/s
            v_opt = np.clip(v_opt, 0.5, 8.0)
            
            # 重新组织数据：[s, x, y, yaw, curvature, v_opt]
            self.raceline_array = np.column_stack([s_vals, x_vals, y_vals, yaw_vals, curvature, v_opt]).astype(np.float32)
            self.get_logger().info(
                f"✅ 已补充缺失列！新 shape={self.raceline_array.shape}\n"
                f" x 范围: [{x_vals.min():.2f}, {x_vals.max():.2f}]\n"
                f" y 范围: [{y_vals.min():.2f}, {y_vals.max():.2f}]\n"
                f" yaw 范围: [{yaw_vals.min():.4f}, {yaw_vals.max():.4f}]\n"
                f" v_opt 范围: [{v_opt.min():.2f}, {v_opt.max():.2f}] m/s"
            )
        else:
            # 已有完整列
            self.raceline_array = raw_array
            x_vals = self.raceline_array[:, 1].astype(np.float64)
            y_vals = self.raceline_array[:, 2].astype(np.float64)
            yaw_vals = self.raceline_array[:, 3].astype(np.float64)
            v_vals = self.raceline_array[:, 5].astype(np.float64)
            s_vals = self.raceline_array[:, 0].astype(np.float64)
            
            segment_lengths = np.hypot(np.diff(x_vals, append=x_vals[0]), np.diff(y_vals, append=y_vals[0]))
            closing_length = float(segment_lengths[-1]) if len(segment_lengths) > 0 else 0.0
            self.raceline_total_length = float(max(np.sum(segment_lengths), s_vals[-1] + closing_length, s_vals[-1]))
            yaw_unwrapped = np.unwrap(yaw_vals)
            closing_yaw = yaw_unwrapped[-1] + self._wrap_angle(yaw_unwrapped[0] - yaw_unwrapped[-1])
            self.raceline_sample_s = np.append(s_vals, self.raceline_total_length)
            self.raceline_sample_x = np.append(x_vals, x_vals[0])
            self.raceline_sample_y = np.append(y_vals, y_vals[0])
            self.raceline_sample_yaw = np.append(yaw_unwrapped, closing_yaw)
            self.raceline_sample_v = np.append(v_vals, v_vals[0])
            
        # 【后续代码不变】
        self.raceline_path_msg = Path()
        self.raceline_path_msg.header.frame_id = 'map'
        for i in range(len(self.raceline_array)):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            x = float(self.raceline_array[i, 1])
            y = float(self.raceline_array[i, 2])
            yaw = float(self.raceline_array[i, 3])
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            
            # 四元数
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            
            self.raceline_path_msg.poses.append(pose)
            
        self.get_logger().info(f"✅ 成功加载赛道线！共 {len(self.raceline_array)} 个点")

    def publish_raceline_timer(self):
        """定时发布 raceline，使其在 RViz2 中持续显示"""
        if self.raceline_path_msg is not None:
            self.raceline_path_msg.header.stamp = self.get_clock().now().to_msg()
            self.raceline_pub.publish(self.raceline_path_msg)

    def _find_closest_raceline_index(self, current_pose):
        if self.raceline_array is None or current_pose is None:
            return 0
        x = float(current_pose.pose.position.x)
        y = float(current_pose.pose.position.y)
        num_points = len(self.raceline_array)
        
        if self.last_raceline_index is None or num_points < 10:
            dx = self.raceline_array[:, 1] - x
            dy = self.raceline_array[:, 2] - y
            best_idx = int(np.argmin(dx * dx + dy * dy))
            self.last_raceline_index = best_idx
            return best_idx
            
        search_radius = int(np.clip(30 + abs(self.odom_speed) * 15.0, 30, max(30, num_points // 3)))
        candidate_offsets = np.arange(-search_radius, search_radius + 1)
        candidate_indices = (self.last_raceline_index + candidate_offsets) % num_points
        dx = self.raceline_array[candidate_indices, 1] - x
        dy = self.raceline_array[candidate_indices, 2] - y
        local_best_pos = int(np.argmin(dx * dx + dy * dy))
        best_idx = int(candidate_indices[local_best_pos])
        best_dist_sq = float(dx[local_best_pos] * dx[local_best_pos] + dy[local_best_pos] * dy[local_best_pos])
        
        if best_dist_sq > 16.0:
            dx = self.raceline_array[:, 1] - x
            dy = self.raceline_array[:, 2] - y
            best_idx = int(np.argmin(dx * dx + dy * dy))
            self.last_raceline_index = best_idx
            
        return best_idx

    def _find_index_by_s(self, target_s):
        if self.raceline_array is None:
            return 0
        s_values = self.raceline_array[:, 0]
        total_length = self.raceline_total_length if self.raceline_total_length > 1e-6 else s_values[-1]
        target_s = target_s % total_length
        idx = np.searchsorted(s_values, target_s)
        if idx == len(s_values):
            idx = 0
        return idx

    def _sample_raceline_by_s(self, target_s):
        if self.raceline_array is None or self.raceline_sample_s is None:
            return None
        total_length = self.raceline_total_length if self.raceline_total_length > 1e-6 else float(self.raceline_array[-1, 0])
        target_s = target_s % total_length
        idx = int(np.searchsorted(self.raceline_sample_s, target_s, side='right') - 1)
        idx = int(np.clip(idx, 0, len(self.raceline_sample_s) - 2))
        s0 = self.raceline_sample_s[idx]
        s1 = self.raceline_sample_s[idx + 1]
        ratio = 0.0 if s1 <= s0 + 1e-6 else float((target_s - s0) / (s1 - s0))
        x = (1.0 - ratio) * self.raceline_sample_x[idx] + ratio * self.raceline_sample_x[idx + 1]
        y = (1.0 - ratio) * self.raceline_sample_y[idx] + ratio * self.raceline_sample_y[idx + 1]
        yaw = self.raceline_sample_yaw[idx] + ratio * (self.raceline_sample_yaw[idx + 1] - self.raceline_sample_yaw[idx])
        v = (1.0 - ratio) * self.raceline_sample_v[idx] + ratio * self.raceline_sample_v[idx + 1]
        return {'x': float(x), 'y': float(y), 'yaw': self._wrap_angle(float(yaw)), 'v': float(v)}

    def _bias_heading_with_laser(self, global_heading, front_state, weight_global=0.85):
        """全局方向 + 激光局部最远点微调（保留反应式避障能力）"""
        max_dir_index = front_state['max_dir_index']
        global_heading = self._wrap_angle(global_heading)
        num_scans = len(front_state['dis_90'])
        global_index = int(np.clip(90 + int(global_heading / math.pi * 180), 0, num_scans - 1))
        window = 40
        best_idx = max_dir_index
        best_dis = 0
        for i in range(max(0, global_index - window), min(num_scans, global_index + window + 1)):
            dis = front_state['dis_90'][i]
            if dis > best_dis:
                best_dis = dis
                best_idx = i
        fused_index = weight_global * global_index + (1 - weight_global) * best_idx
        return (fused_index - 90) / 180.0 * math.pi

    def build_local_ref(self, front_state, target_speed, speed_upper_bound, current_pose=None):
        """全局 Raceline + 激光反应式融合版"""
        current_pose = current_pose if current_pose is not None else self.odom_pose
        if self.raceline_array is None or current_pose is None:
            # 没有 raceline → 退回你原来的纯反应式
            target_heading = (float(front_state['max_dir_index']) - 90.0) / 180.0 * math.pi
            target_heading = float(np.clip(target_heading, -0.9, 0.9))
            v_des = float(np.clip(target_speed, max(self.mpc_min_speed, 0.0), speed_upper_bound))
            x_ref = np.zeros(self.mpc_horizon)
            y_ref = np.zeros(self.mpc_horizon)
            yaw_ref = np.zeros(self.mpc_horizon)
            v_ref = np.zeros(self.mpc_horizon)
            for k in range(self.mpc_horizon):
                s = (k + 1) * self.mpc_dt * max(v_des, 0.3)
                x_ref[k] = s * math.cos(target_heading)
                y_ref[k] = s * math.sin(target_heading)
                yaw_ref[k] = target_heading
                v_ref[k] = v_des
            return {'x': x_ref, 'y': y_ref, 'yaw': yaw_ref, 'v': v_ref}
        else:
            # 使用 Raceline 生成车体系下的局部参考轨迹
            closest_idx = self._find_closest_raceline_index(current_pose)
            current_s = float(self.raceline_array[closest_idx, 0])
            current_x = float(current_pose.pose.position.x)
            current_y = float(current_pose.pose.position.y)
            current_yaw = self._pose_yaw(current_pose)
            x_ref = np.zeros(self.mpc_horizon)
            y_ref = np.zeros(self.mpc_horizon)
            yaw_ref = np.zeros(self.mpc_horizon)
            v_ref = np.zeros(self.mpc_horizon)
            preview_s = current_s + max(self.pp_lookahead, 0.4 + 0.35 * abs(self.odom_speed))
            sample_s = preview_s
            for k in range(self.mpc_horizon):
                sample = self._sample_raceline_by_s(sample_s)
                if sample is None:
                    break
                dx = sample['x'] - current_x
                dy = sample['y'] - current_y
                local_x, local_y = self._world_to_body(dx, dy, current_yaw)
                x_ref[k] = local_x
                y_ref[k] = local_y
                yaw_ref[k] = self._wrap_angle(sample['yaw'] - current_yaw)
                v_ref[k] = float(np.clip(sample['v'], self.mpc_min_speed, speed_upper_bound))
                sample_s += max(v_ref[k], 0.5) * self.mpc_dt
                
            if np.hypot(x_ref[0], y_ref[0]) < 1e-6:
                raceline_heading = yaw_ref[0]
            else:
                raceline_heading = math.atan2(y_ref[0], x_ref[0])
                
            laser_heading = float(np.clip((float(front_state['max_dir_index']) - 90.0) / 180.0 * math.pi, -0.9, 0.9))
            weight_global = self._compute_raceline_blend_weight(front_state)
            fused_heading = self._blend_angle(laser_heading, raceline_heading, weight_global)
            # 【修复】移除_bias_heading_with_laser二次覆盖，该函数会把融合航向拉回激光最远点方向，
            # 在弯道处导致车辆不跟弯。一次融合已经包含了激光信息，无需二次修正。
            blend_steps = min(2, self.mpc_horizon)  # 【修复】从4改为2，减少对Raceline参考点的修改范围
            
            self.get_logger().debug(
                f"Raceline跟踪: idx={closest_idx}, w={weight_global:.2f}, "
                f"rl_h={raceline_heading:.3f}, laser_h={laser_heading:.3f}, fused={fused_heading:.3f}"
            )
            
            for k in range(blend_steps):
                distance_k = float(np.hypot(x_ref[k], y_ref[k]))
                if distance_k < 1e-6:
                    continue
                nominal_heading = math.atan2(y_ref[k], x_ref[k])
                blend_ratio = float(blend_steps - k) / float(blend_steps)
                heading_k = self._blend_angle(nominal_heading, fused_heading, blend_ratio)
                x_ref[k] = distance_k * math.cos(heading_k)
                y_ref[k] = distance_k * math.sin(heading_k)
                yaw_ref[k] = heading_k
                
            if np.all(v_ref <= 0.0):
                v_ref[:] = float(np.clip(target_speed, self.mpc_min_speed, speed_upper_bound))
            return {'x': x_ref, 'y': y_ref, 'yaw': yaw_ref, 'v': v_ref}

    # ====================== 下面所有函数保持不变（只改了调用处） ======================
    def rollout_cost(self, steer, accel, state, ref, speed_upper_bound):
        x = 0.0
        y = 0.0
        yaw = 0.0
        v = float(state['v'])
        cost = 0.0
        for k in range(self.mpc_horizon):
            x += v * math.cos(yaw) * self.mpc_dt
            y += v * math.sin(yaw) * self.mpc_dt
            yaw += v / self.wheelbase * math.tan(steer) * self.mpc_dt
            v = np.clip(v + accel * self.mpc_dt, self.mpc_min_speed, speed_upper_bound)
            ex = x - ref['x'][k]
            ey = y - ref['y'][k]
            eyaw = (yaw - ref['yaw'][k] + math.pi) % (2 * math.pi) - math.pi
            ev = v - ref['v'][k]
            cost += 2.0 * ex * ex + 3.0 * ey * ey + 0.8 * eyaw * eyaw + 0.3 * ev * ev
            cost += 0.05 * steer * steer + 0.02 * accel * accel
            cost += 0.12 * (steer - self.last_mpc_steer) * (steer - self.last_mpc_steer)
        return float(cost), float(v)

    def solve_mpc(self, heuristic, ref, speed_upper_bound):
        if not self.have_odom or ref is None:
            return False, heuristic['steering'], heuristic['speed'], 'no_odom', 0.0
            
        start_t = time.monotonic()
        timeout_s = self.mpc_timeout_ms / 1000.0
        steer_grid = np.linspace(-math.pi / 4, math.pi / 4, self.mpc_steer_candidates)
        accel_grid = [0.0]
        
        if self.mpc_mode == 'full':
            accel_grid = np.linspace(self.mpc_max_decel, self.mpc_max_accel, self.mpc_accel_candidates)
            
        state = {'v': float(abs(self.odom_speed))}
        best_cost = float('inf')
        best_steer = heuristic['steering']
        best_speed = heuristic['speed']
        
        for steer in steer_grid:
            for accel in accel_grid:
                if time.monotonic() - start_t > timeout_s:
                    elapsed_ms = (time.monotonic() - start_t) * 1000.0
                    return False, heuristic['steering'], heuristic['speed'], 'timeout', elapsed_ms
                    
                cost, vend = self.rollout_cost(steer=float(steer), accel=float(accel), state=state, ref=ref, speed_upper_bound=speed_upper_bound)
                
                if cost < best_cost:
                    best_cost = cost
                    best_steer = float(steer)
                    if self.mpc_mode == 'full':
                        best_speed = float(np.clip(vend, self.mpc_min_speed, speed_upper_bound))
                        
        self.last_mpc_steer = best_steer
        self.last_mpc_speed = best_speed
        elapsed_ms = (time.monotonic() - start_t) * 1000.0
        return True, best_steer, best_speed, 'ok', elapsed_ms

    def pure_pursuit_speed(self, ref, base_speed):
        lookahead = self.pp_lookahead
        x = np.array(ref['x'])
        y = np.array(ref['y'])
        dists = np.sqrt(x**2 + y**2)
        idxs = np.where(dists >= lookahead)[0]
        
        if len(idxs) == 0:
            idx = len(dists) - 1
        else:
            idx = idxs[0]
            
        yg = y[idx]
        ld = dists[idx]
        if ld < 1e-6:
            return float(base_speed)
            
        curvature = 2.0 * yg / (ld ** 2)
        # PP弯道衰减系数：从0.8降低到0.35，减少弯道过度减速
        speed_factor = 1.0 / (1.0 + 0.35 * abs(curvature) * self.wheelbase)
        speed = base_speed * speed_factor
        speed = float(np.clip(speed, self.mpc_min_speed, self.final_speed_cap))
        return speed

    def control_select_and_publish(self, data, current_frame, front_state, heuristic, current_pose=None):
        drive_msg = AckermannDriveStamped()
        drive_msg.header = data.header
        
        heuristic_msg = AckermannDriveStamped()
        heuristic_msg.header = data.header
        heuristic_msg.drive.steering_angle = float(heuristic['steering'])
        heuristic_msg.drive.speed = float(heuristic['speed'])
        
        mode = CONTROL_MODE_HEURISTIC
        steer_cmd = heuristic['steering']
        speed_cmd = heuristic['speed']
        mpc_ok = False
        mpc_solve_ms = 0.0
        self.last_mpc_reason = 'inactive'
        self.control_cycle_count += 1
        
        # === 模式状态机更新（平滑切换，替代原bool硬切换） ===
        driving_mode = self.mode_mgr.update(
            self.dynamic_obs, self.Follow, self.chaoche,
            len(front_state['left_obs']) > 0, self.mpc_dt
        )
        speed_upper_bound = min(self.mode_mgr.speed_upper_bound, self.final_speed_cap)
        self.current_speed_upper_bound = speed_upper_bound
        
        # === 混合模式：向量化MPC转向 + Pure Pursuit速度 ===
        ref = None
        if self.pure_pursuit_enable and self.mpc_enable and self.mpc_mode in ('steer_only', 'full'):
            self.mpc_attempt_count += 1
            ref = self.build_local_ref(front_state, heuristic['speed'], speed_upper_bound, current_pose)
            
            # 使用向量化MPC求解器（含动力学模型）
            mpc_ok, mpc_steer, mpc_solve_ms = self.mpc_solver.solve(
                abs(self.odom_speed), ref, speed_upper_bound
            )
            
            steer_cmd = mpc_steer if mpc_ok else heuristic['steering']
            pp_base_speed = heuristic['speed']
            
            if ref is not None and len(ref['v']) > 0:
                pp_base_speed = float(min(pp_base_speed, ref['v'][0]))
            pp_speed = self.pure_pursuit_speed(ref, pp_base_speed)
            speed_cmd = pp_speed
            mode = CONTROL_MODE_MPC_PP if mpc_ok else CONTROL_MODE_HEURISTIC
            
        self.last_solve_time_ms = float(mpc_solve_ms)
        
        if mpc_ok:
            self.mpc_success_count += 1
            self.sum_solve_time_ms += float(mpc_solve_ms)
            self.max_solve_time_ms = max(self.max_solve_time_ms, float(mpc_solve_ms))
            
        self.last_mpc_steer = steer_cmd
        self.last_mpc_speed = speed_cmd
        speed_cmd = float(min(speed_cmd, speed_upper_bound))
        
        # 模式速度约束（使用平滑后的上限）
        if driving_mode == DrivingMode.FOLLOW:
            self.get_logger().debug('动态跟随模式')
        elif driving_mode == DrivingMode.OVERTAKE:
            self.get_logger().info('超车模式启动')
            
        drive_msg.drive.steering_angle = float(np.clip(steer_cmd, -math.pi / 4, math.pi / 4))
        drive_msg.drive.speed = float(np.clip(speed_cmd, 0.0, self.final_speed_cap))
        
        self.publish_arrow_marker(front_state['max_dir_index'], current_frame)
        self.last_max_dir_index = front_state['max_dir_index']
        
        mpc_msg = AckermannDriveStamped()
        mpc_msg.header = data.header
        mpc_msg.drive.steering_angle = float(np.clip(self.last_mpc_steer, -math.pi / 4, math.pi / 4))
        mpc_msg.drive.speed = float(np.clip(self.last_mpc_speed, 0.0, self.mpc_max_speed))
        
        self.heuristic_pub.publish(heuristic_msg)
        self.mpc_pub.publish(mpc_msg)
        
        mode_msg = UInt8()
        mode_msg.data = int(driving_mode)
        self.mode_pub.publish(mode_msg)
        
        ok_msg = Bool()
        ok_msg.data = bool(mpc_ok)
        self.mpc_ok_pub.publish(ok_msg)
        
        self.last_control_mode = mode
        self.pre_safety_pub.publish(drive_msg)
        self.drive_pub.publish(drive_msg)
        
        # === 实验数据记录 ===
        blend_w = self._compute_raceline_blend_weight(front_state)
        front_window = front_state['dis_90'][80:101]
        
        # 计算横向跟踪误差（如有Raceline和ref）
        lat_err = 0.0
        if ref is not None and len(ref['y']) > 0:
            lat_err = float(ref['y'][0])  # 最近参考点的横向偏差即为横向误差
        
        record = CycleRecord(
            mode=int(driving_mode),
            mode_name=self.mode_mgr.get_mode_name(),
            steer=drive_msg.drive.steering_angle,
            speed=drive_msg.drive.speed,
            actual_speed=self.odom_speed,
            speed_upper_bound=speed_upper_bound,
            mpc_ok=mpc_ok,
            mpc_ms=mpc_solve_ms,
            blend_weight=blend_w,
            front_clearance=float(np.min(front_window)) if len(front_window) > 0 else 0.0,
            num_obstacles=len(front_state['left_obs']) // 2,
            lateral_error=lat_err,
        )
        if self.odom_pose:
            record.pos_x = self.odom_pose.pose.position.x
            record.pos_y = self.odom_pose.pose.position.y
            record.heading = self.odom_yaw
        self.exp_logger.log(record)

    def middle_line_callback(self, data):
        clean_ranges = np.nan_to_num(np.array(data.ranges), posinf=0.0, neginf=0.0)
        data.ranges = clean_ranges.tolist()
        current_frame = data.header.frame_id if data.header.frame_id else 'laser'
        front_state = self.frontend_scan_process(data)
        heuristic = self.heuristic_backend_control(front_state)
        self.control_select_and_publish(data, current_frame, front_state, heuristic, self.odom_pose)


def main(args=None):
    rclpy.init(args=args)
    battle_node = BattleVehicleNode()
    try:
        rclpy.spin(battle_node)
    except KeyboardInterrupt:
        battle_node.get_logger().info('\n正在保存实验数据...')
        report = battle_node.exp_logger.save_all()
        # 在终端直接打印完整报告
        print('\n' + report)
        battle_node.get_logger().info('数据已保存，详见上方报告')
    finally:
        battle_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()