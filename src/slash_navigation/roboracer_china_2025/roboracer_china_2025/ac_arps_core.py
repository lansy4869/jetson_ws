"""AC-ARPS：自适应保形 Ackermann 残差预测安全层核心。

Adaptive Conformal Ackermann Residual Predictive Shield (AC-ARPS)

与 battle_fast2 基础控制器的关系（残差式预测安全过滤器）：
- 本模块**不做方向决策**，只回答一个问题：
  「基础控制器当前准备发布的 (v_base, δ_base)，按 Ackermann 自行车模型
   外推 0.4~0.9 s，是否安全？」
- 判定安全时完全透传，速度和转角一位小数都不改（第 0 级）；
- 不安全时优先只按比例减速、保持 δ_base 不变（第 1 级）；
- 减速仍不安全时才叠加 |Δδ| ≤ max_steering_residual（默认约 6°）的
  小幅转角残差（第 2 级）；
- 所有候选都不安全时输出紧急制动 v=0（第 3 级），转角取已评估候选中
  预测净空最大的一个（"最不坏"的停车轨迹）。

三个可分别消融的机制（对应论文三个创新点）：
1. Ackermann 短时轨迹预测 + 三圆车体包络（几何净空评估）；
2. 连续两帧径向 closing speed 的保守型方向性 TTC（时间维风险）；
3. 「预测净空高于实际净空」危险侧残差的在线保形余量 q_t
   （Adaptive Conformal 思路的在线安全边界校准）。

运动学模型（离散 Ackermann 自行车模型，位姿参考点取后轴中心）：

    x_{k+1}   = x_k + v_k * cos(psi_k) * dt
    y_{k+1}   = y_k + v_k * sin(psi_k) * dt
    psi_{k+1} = psi_k + v_k / L * tan(delta_k) * dt

其中 v 为车速、delta 为前轮转角、L 为轴距。候选指令并非瞬间生效：
前 tau_delay 秒维持"上一帧实际发布"的 (v, δ)（控制/底盘延迟），
之后速度以 a_brake / a_accel 的斜率向候选值逼近、转角切换为候选值。
这使得"v=0 候选"评估的是真实的**制动包络**而不是瞬间停车，
第 2 级转角残差（边刹边微调方向）也因此有意义。

本文件刻意不依赖 ROS，只用 numpy + 标准库，方便离线单元测试与复用。
"""

import math
import time
from collections import deque

import numpy as np

# 介入级别常量
LEVEL_PASS = 0        # 原命令透传
LEVEL_SPEED = 1       # 仅减速
LEVEL_RESIDUAL = 2    # 减速 + 小幅转角残差
LEVEL_EMERGENCY = 3   # 紧急制动（无安全候选）

# 调试数组字段顺序（供 Float32MultiArray 使用，与 debug_to_array() 一致）
DEBUG_ARRAY_FIELDS = (
    "intervention_level",
    "base_speed",
    "suggested_speed",
    "base_steering",
    "suggested_steering",
    "applied",
    "predicted_clearance",
    "ttc",
    "conformal_margin",
    "effective_radius",
    "compute_time_ms",
    "num_points",
)


def debug_to_array(debug):
    """把 filter() 返回的 debug dict 压成固定顺序的 float 列表（发调试话题用）。"""
    out = []
    for key in DEBUG_ARRAY_FIELDS:
        value = float(debug.get(key, 0.0))
        if not math.isfinite(value):
            # inf 的 TTC / 净空压到一个大数，避免 Float32MultiArray 出 inf
            value = 1.0e6 if value > 0 else -1.0e6
        out.append(value)
    return out


class AdaptiveConformalAckermannShield:
    """自适应保形 Ackermann 残差预测安全层。

    每帧调用一次 filter()。类内部维护三份跨帧状态：
    - 上一帧实际发布的 (v, δ)（延迟建模用）；
    - 上一帧原始扫描 + 时间戳（TTC 用）；
    - 保形残差窗口 + 上一帧对"本帧净空"的预测值（在线余量用）。
    """

    def __init__(
        self,
        # --- 车辆几何（务必按实车实测修改）---
        wheelbase=0.33,             # 轴距 L [m]（注意 vesc.yaml 里记录为 0.25，上车前核实）
        vehicle_length=0.50,        # 含前后悬的车身总长 [m]
        vehicle_width=0.30,         # 车宽 [m]
        laser_x_offset=0.12,        # 雷达原点在后轴前方的纵向偏移 [m]
        # --- 预测 ---
        prediction_dt=0.05,         # 预测步长 [s]
        min_horizon=0.40,           # 预测时域下限 [s]
        max_horizon=0.85,           # 预测时域上限 [s]
        tau_delay=0.16,             # 控制+底盘延迟 [s]（与 reachability_system_delay 对齐）
        a_brake=2.5,                # 制动减速度 [m/s^2]
        a_accel=2.0,                # 加速斜率 [m/s^2]（延迟建模用）
        # --- 分级介入 ---
        speed_scales=(0.85, 0.70, 0.50, 0.25, 0.0),   # 第 1 级速度比例，从高到低
        residual_step=0.035,        # 转角残差步长 [rad]
        max_steering_residual=0.105,  # 最大转角残差 [rad] ≈ 6°
        # --- 安全余量 ---
        fixed_margin=0.08,          # 固定安全余量 [m]
        adaptive_margin_enabled=False,  # 在线保形余量开关（阶段 3 再打开）
        target_miscoverage=0.05,    # 保形失覆盖率 alpha
        conformal_window=200,       # 残差滑动窗口帧数
        conformal_min_samples=30,   # 少于该样本数时 q_t 记 0
        max_conformal_margin=0.30,  # q_t 上限 [m]，防止余量爆炸
        # --- TTC ---
        ttc_emergency=0.35,         # TTC 低于此值 → 速度上限压到 0 [s]
        ttc_slow=1.20,              # TTC 低于此值开始线性限速 [s]
        ttc_percentile=20.0,        # 扇区内 TTC 取的分位数（抗单点噪声）
        ttc_sector_half_width=0.35,  # 以未来行进方向为中心的扇区半宽 [rad]
        ttc_min_closing=0.10,       # 视为"接近中"的最小径向速度 [m/s]
        ttc_min_points=5,           # 少于该数量的接近点则本帧 TTC 记 inf
        # --- 扫描预处理 ---
        max_scan_points=90,         # 角度分桶取最近点后的目标点数
        roi_half_angle=2.4,         # 参与碰撞检查的点的方位角半宽 [rad]（后轴系）
        roi_radius=6.0,             # 参与碰撞检查的点的最大距离 [m]
        min_valid_beams=10,         # 有效+确认空旷 beam 少于此数 → 感知降级
        degraded_speed_cap=0.5,     # 感知降级时的速度上限 [m/s]
        # --- 输出限幅与模式 ---
        max_speed=2.7,              # 输出速度上限 [m/s]
        max_steer=math.pi / 4.0,    # 输出转角限幅 [rad]
        shadow_mode=True,           # 影子模式：只记录建议，不修改输出（阶段 0）
        speed_only=True,            # 只允许减速，不允许转角残差（阶段 1）
    ):
        if vehicle_length <= 0.0 or vehicle_width <= 0.0 or wheelbase <= 0.0:
            raise ValueError("vehicle geometry must be positive")
        self.wheelbase = float(wheelbase)
        self.vehicle_length = float(vehicle_length)
        self.vehicle_width = float(vehicle_width)
        self.laser_x_offset = float(laser_x_offset)

        self.prediction_dt = float(prediction_dt)
        self.min_horizon = float(min_horizon)
        self.max_horizon = float(max_horizon)
        self.tau_delay = float(tau_delay)
        self.a_brake = max(float(a_brake), 1e-3)
        self.a_accel = max(float(a_accel), 1e-3)

        self.speed_scales = tuple(float(s) for s in speed_scales)
        self.residual_step = float(residual_step)
        self.max_steering_residual = float(max_steering_residual)

        self.fixed_margin = float(fixed_margin)
        self.adaptive_margin_enabled = bool(adaptive_margin_enabled)
        self.target_miscoverage = float(np.clip(target_miscoverage, 0.001, 0.5))
        self.conformal_min_samples = int(conformal_min_samples)
        self.max_conformal_margin = float(max_conformal_margin)

        self.ttc_emergency = float(ttc_emergency)
        self.ttc_slow = max(float(ttc_slow), self.ttc_emergency + 1e-3)
        self.ttc_percentile = float(ttc_percentile)
        self.ttc_sector_half_width = float(ttc_sector_half_width)
        self.ttc_min_closing = float(ttc_min_closing)
        self.ttc_min_points = int(ttc_min_points)

        self.max_scan_points = max(int(max_scan_points), 8)
        self.roi_half_angle = float(roi_half_angle)
        self.roi_radius = float(roi_radius)
        self.min_valid_beams = int(min_valid_beams)
        self.degraded_speed_cap = float(degraded_speed_cap)

        self.max_speed = float(max_speed)
        self.max_steer = float(max_steer)
        self.shadow_mode = bool(shadow_mode)
        self.speed_only = bool(speed_only)

        # --- 三圆车体包络（完整覆盖 车长 x 车宽 矩形）---
        # 后悬 = (车长 - 轴距)/2（车身相对轴距居中的近似，可按实车再调）
        rear_overhang = max((self.vehicle_length - self.wheelbase) / 2.0, 0.0)
        seg = self.vehicle_length / 6.0
        # 三个圆心沿车身纵轴均布（后轴系坐标），半径取矩形覆盖圆半径
        self.circle_offsets = np.array(
            [-rear_overhang + seg, -rear_overhang + 3.0 * seg, -rear_overhang + 5.0 * seg],
            dtype=np.float64,
        )
        self.circle_radius_geom = math.hypot(seg, self.vehicle_width / 2.0)
        # TTC 用：车头最前端到雷达原点的纵向距离（径向 TTC 的距离修正量）
        self.front_reach_from_laser = (
            float(self.circle_offsets[-1]) + self.circle_radius_geom - self.laser_x_offset
        )

        # --- 跨帧状态 ---
        self._last_cmd_speed = 0.0        # 上一帧实际发布的速度（延迟建模）
        self._last_cmd_steer = 0.0        # 上一帧实际发布的转角
        self._prev_ranges = None          # 上一帧原始 ranges（TTC）
        self._prev_valid = None
        self._prev_stamp = None
        self._frame_dt = 0.05             # 帧间隔滑动估计 [s]
        self._scores = deque(maxlen=int(conformal_window))  # 保形残差窗口
        self._pred_next_clearance = None  # 上一帧对"本帧实际净空"的预测值
        self.frames_processed = 0

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def prime(self, speed, steering):
        """用外部信息（如里程计/上一条指令）初始化延迟建模的当前指令估计。"""
        self._last_cmd_speed = float(speed)
        self._last_cmd_steer = float(steering)

    @property
    def conformal_margin(self):
        """当前在线保形余量 q_t [m]（未启用或样本不足时为 0）。"""
        if not self.adaptive_margin_enabled:
            return 0.0
        if len(self._scores) < max(self.conformal_min_samples, 2):
            return 0.0
        q = float(np.quantile(np.asarray(self._scores), 1.0 - self.target_miscoverage))
        return float(np.clip(q, 0.0, self.max_conformal_margin))

    @property
    def effective_radius(self):
        """当前有效安全半径 = 三圆几何半径 + 固定余量 + 保形余量 [m]。"""
        return self.circle_radius_geom + self.fixed_margin + self.conformal_margin

    def filter(
        self,
        ranges,
        angle_min,
        angle_increment,
        range_min,
        range_max,
        base_speed,
        base_steering,
        stamp_sec,
    ):
        """对基础控制器的 (base_speed, base_steering) 做预测安全过滤。

        必须用**未经 nan_to_num / 补零处理的原始 ranges** 调用；
        inf 表示量程内无回波（自由空间），NaN 表示无效测量，两者语义不同。

        返回 (safe_speed, safe_steering, debug_dict)。
        shadow_mode 下返回值恒等于基础指令，建议值写在 debug 里。
        """
        t_start = time.perf_counter()
        base_speed = float(base_speed)
        base_steering = float(base_steering)

        # ---- 1. 帧间隔估计 ----
        stamp_sec = float(stamp_sec)
        if self._prev_stamp is not None:
            dt = stamp_sec - self._prev_stamp
            if 1e-3 < dt < 0.5:
                self._frame_dt = 0.8 * self._frame_dt + 0.2 * dt
            elif not (0.0 < dt < 2.0):
                # 时间戳异常跳变（如 rosbag 回放循环）：重置跨帧状态
                self._reset_temporal_state()

        # ---- 2. 扫描预处理（原始 inf/NaN 语义保留）----
        r_raw = np.asarray(ranges, dtype=np.float64).ravel()
        points, n_info, angles = self._preprocess_scan(
            r_raw, angle_min, angle_increment, range_min, range_max
        )
        degraded = n_info < self.min_valid_beams

        # ---- 3. 当前有效安全半径（固定余量 + 保形余量）----
        q_t = self.conformal_margin
        r_eff = self.circle_radius_geom + self.fixed_margin + q_t

        # 当前位姿的实际观测净空（保形分数用，不含任何余量）
        clearance_now = self._pose_clearance(points)

        # ---- 4. 倒车 / 停车指令直接透传（本层只保护前向运动）----
        if base_speed < 0.05:
            ttc_dir = self._update_ttc(r_raw, angle_min, angle_increment,
                                       range_min, range_max, stamp_sec,
                                       direction=0.0)
            self._finish_frame(
                executed_speed=base_speed, executed_steer=base_steering,
                pred_pose=(0.0, 0.0, 0.0), points=points,
                clearance_now=clearance_now, stamp_sec=stamp_sec,
            )
            debug = self._make_debug(
                level=LEVEL_PASS, reason="base_stop_or_reverse",
                base_speed=base_speed, base_steering=base_steering,
                suggested_speed=base_speed, suggested_steering=base_steering,
                c_min=clearance_now - r_eff, ttc=ttc_dir, q_t=q_t, r_eff=r_eff,
                n_points=points.shape[0], t_start=t_start, degraded=degraded,
            )
            return base_speed, base_steering, debug

        # ---- 5. 预测时域 H(v) = clip(tau + v/a_brake, min, max) ----
        v_ref = max(base_speed, self._last_cmd_speed)
        horizon = float(np.clip(
            self.tau_delay + v_ref / self.a_brake, self.min_horizon, self.max_horizon
        ))
        n_steps = max(int(math.ceil(horizon / self.prediction_dt)), 2)

        # ---- 6. 第 0 级：评估基础指令本身 ----
        c_base, poses_base = self._evaluate_candidates(
            np.array([base_speed]), np.array([base_steering]), points, r_eff, n_steps
        )
        c_base = float(c_base[0])
        base_poses = poses_base[0]

        # ---- 7. 方向性保守 TTC（以基础轨迹末端方向为扇区中心）----
        end_x, end_y = base_poses[-1, 0], base_poses[-1, 1]
        direction = math.atan2(end_y, end_x) if (end_x * end_x + end_y * end_y) > 1e-6 else 0.0
        ttc_dir = self._update_ttc(r_raw, angle_min, angle_increment,
                                   range_min, range_max, stamp_sec,
                                   direction=direction)

        # TTC → 速度上限：ttc >= slow 不限速；<= emergency 压到 0；中间线性
        if ttc_dir >= self.ttc_slow:
            ttc_cap = self.max_speed
        elif ttc_dir <= self.ttc_emergency:
            ttc_cap = 0.0
        else:
            ratio = (ttc_dir - self.ttc_emergency) / (self.ttc_slow - self.ttc_emergency)
            ttc_cap = base_speed * ratio
        speed_cap = min(self.max_speed, ttc_cap)
        if degraded:
            speed_cap = min(speed_cap, self.degraded_speed_cap)

        # ---- 8. 分级决策 ----
        level = LEVEL_PASS
        reason = "pass_through"
        chosen_speed, chosen_steer = base_speed, base_steering
        chosen_c, chosen_poses = c_base, base_poses

        base_clear_ok = c_base > 0.0
        base_speed_ok = base_speed <= speed_cap + 1e-9

        if not (base_clear_ok and base_speed_ok):
            # 记录第 0 级失败原因（论文统计介入原因用）
            fail = []
            if not base_clear_ok:
                fail.append("clearance")
            if not base_speed_ok:
                fail.append("ttc" if ttc_cap < base_speed else "degraded")
            fail_reason = "+".join(fail)

            found = False
            # --- 第 1 级：只减速，保持 δ_base ---
            v_cands = np.array([s * base_speed for s in self.speed_scales])
            d_cands = np.full(v_cands.shape, base_steering)
            c_l1, poses_l1 = self._evaluate_candidates(v_cands, d_cands, points, r_eff, n_steps)
            for i in range(v_cands.size):
                if c_l1[i] > 0.0 and v_cands[i] <= speed_cap + 1e-9:
                    level = LEVEL_SPEED
                    reason = "speed_only:" + fail_reason
                    chosen_speed = float(v_cands[i])
                    chosen_steer = base_steering
                    chosen_c, chosen_poses = float(c_l1[i]), poses_l1[i]
                    found = True
                    break

            # --- 第 2 级：小幅转角残差（按 |Δδ| 从小到大，同残差内速度从高到低）---
            if not found and not self.speed_only:
                residuals = self._residual_list()
                l3_scales = (1.0,) + self.speed_scales
                v_list, d_list = [], []
                for res in residuals:
                    d = float(np.clip(base_steering + res, -self.max_steer, self.max_steer))
                    for s in l3_scales:
                        v_list.append(s * base_speed)
                        d_list.append(d)
                v_arr = np.asarray(v_list)
                d_arr = np.asarray(d_list)
                c_l2, poses_l2 = self._evaluate_candidates(v_arr, d_arr, points, r_eff, n_steps)
                for i in range(v_arr.size):
                    if c_l2[i] > 0.0 and v_arr[i] <= speed_cap + 1e-9:
                        level = LEVEL_RESIDUAL
                        reason = "steer_residual:" + fail_reason
                        chosen_speed = float(v_arr[i])
                        chosen_steer = float(d_arr[i])
                        chosen_c, chosen_poses = float(c_l2[i]), poses_l2[i]
                        found = True
                        break
                # 紧急兜底备选：所有 v=0 候选里预测净空最大的转角
                if not found:
                    zero_mask = v_arr < 1e-9
                    if np.any(zero_mask):
                        idx = int(np.argmax(np.where(zero_mask, c_l2, -np.inf)))
                        best_stop_steer = float(d_arr[idx])
                        best_stop_c = float(c_l2[idx])
                        best_stop_poses = poses_l2[idx]
                    else:
                        best_stop_steer, best_stop_c, best_stop_poses = (
                            base_steering, float(c_l1[-1]), poses_l1[-1]
                        )
            else:
                # speed_only 模式的紧急兜底：v=0 + δ_base（第 1 级最后一个候选）
                best_stop_steer = base_steering
                best_stop_c = float(c_l1[-1])
                best_stop_poses = poses_l1[-1]

            # --- 第 3 级：紧急制动 ---
            if not found:
                level = LEVEL_EMERGENCY
                reason = "emergency_brake:" + fail_reason
                chosen_speed = 0.0
                chosen_steer = best_stop_steer
                chosen_c, chosen_poses = best_stop_c, best_stop_poses

        # ---- 9. 输出限幅 ----
        chosen_speed = float(np.clip(chosen_speed, 0.0, self.max_speed))
        chosen_steer = float(np.clip(chosen_steer, -self.max_steer, self.max_steer))

        # ---- 10. 影子模式：只记录建议，输出保持基础指令 ----
        if self.shadow_mode:
            out_speed, out_steer = base_speed, base_steering
            executed_poses = base_poses
        else:
            out_speed, out_steer = chosen_speed, chosen_steer
            executed_poses = chosen_poses

        # ---- 11. 保形状态更新（用"实际会被执行"的轨迹预测下一帧净空）----
        pred_idx = int(np.clip(round(self._frame_dt / self.prediction_dt) - 1,
                               0, executed_poses.shape[0] - 1))
        pred_pose = (
            float(executed_poses[pred_idx, 0]),
            float(executed_poses[pred_idx, 1]),
            float(executed_poses[pred_idx, 2]),
        )
        self._finish_frame(
            executed_speed=out_speed, executed_steer=out_steer,
            pred_pose=pred_pose, points=points,
            clearance_now=clearance_now, stamp_sec=stamp_sec,
        )

        debug = self._make_debug(
            level=level, reason=reason,
            base_speed=base_speed, base_steering=base_steering,
            suggested_speed=chosen_speed, suggested_steering=chosen_steer,
            c_min=chosen_c, ttc=ttc_dir, q_t=q_t, r_eff=r_eff,
            n_points=points.shape[0], t_start=t_start, degraded=degraded,
        )
        return out_speed, out_steer, debug

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _reset_temporal_state(self):
        """时间戳异常时重置所有依赖帧间连续性的状态。"""
        self._prev_ranges = None
        self._prev_valid = None
        self._prev_stamp = None
        self._pred_next_clearance = None

    def _residual_list(self):
        """生成转角残差序列：(-step, +step, -2*step, +2*step, ...)。"""
        residuals = []
        n = int(round(self.max_steering_residual / max(self.residual_step, 1e-6)))
        for k in range(1, n + 1):
            residuals.append(-k * self.residual_step)
            residuals.append(+k * self.residual_step)
        return residuals

    def _preprocess_scan(self, r_raw, angle_min, angle_increment, range_min, range_max):
        """原始扫描 → 后轴系障碍点集（角度分桶保留每桶最近点）。

        - NaN / 低于 range_min：无效测量，直接剔除（不参与碰撞，也不算空旷）；
        - inf / 达到 range_max：量程内无回波，视为自由空间（不生成障碍点）；
        - 其余为有效回波，变换到后轴坐标系后按方位角分桶取最近点降采样。
        """
        n = r_raw.size
        if n == 0 or angle_increment == 0.0:
            return np.empty((0, 2)), 0, np.empty(0)

        angles = angle_min + np.arange(n, dtype=np.float64) * angle_increment
        finite = np.isfinite(r_raw)
        lower = max(float(range_min), 0.05)
        upper = float(range_max) * 0.999 if range_max > 0 else np.inf
        valid = finite & (r_raw > lower) & (r_raw < upper)
        free = np.isposinf(r_raw) | (finite & (r_raw >= upper) & (r_raw > 0))
        n_info = int(np.count_nonzero(valid) + np.count_nonzero(free))

        if not np.any(valid):
            return np.empty((0, 2)), n_info, angles

        r_v = r_raw[valid]
        a_v = angles[valid]
        # 雷达系 → 后轴系（雷达在后轴前方 laser_x_offset 处，无旋转）
        px = r_v * np.cos(a_v) + self.laser_x_offset
        py = r_v * np.sin(a_v)
        pr = np.hypot(px, py)
        pa = np.arctan2(py, px)

        roi = (np.abs(pa) <= self.roi_half_angle) & (pr <= self.roi_radius)
        if not np.any(roi):
            return np.empty((0, 2)), n_info, angles
        px, py, pr, pa = px[roi], py[roi], pr[roi], pa[roi]

        if px.size <= self.max_scan_points:
            return np.column_stack((px, py)), n_info, angles

        # 角度分桶，每桶保留距离最近的原始点（保守降采样，不产生量化位移）
        span = 2.0 * self.roi_half_angle
        bucket = np.clip(
            ((pa + self.roi_half_angle) / span * self.max_scan_points).astype(np.int64),
            0, self.max_scan_points - 1,
        )
        order = np.lexsort((pr, bucket))
        _, first_idx = np.unique(bucket[order], return_index=True)
        keep = order[first_idx]
        return np.column_stack((px[keep], py[keep])), n_info, angles

    def _control_profiles(self, v_cmds, d_cmds, n_steps):
        """构造各候选的 (速度, 转角) 时间剖面，包含控制延迟与加减速斜率。

        延迟窗口 [0, tau_delay) 内维持上一帧实际发布的指令；
        之后速度按 a_brake / a_accel 向候选值逼近（闭式解），转角切换为候选值。
        """
        dt = self.prediction_dt
        times = np.arange(n_steps, dtype=np.float64) * dt  # 每步起始时刻
        elapsed = np.maximum(times - self.tau_delay, 0.0)  # 延迟后的经过时间

        v0 = self._last_cmd_speed
        d0 = self._last_cmd_steer
        dv_target = v_cmds[:, None] - v0  # (n_cand, 1)
        dv = np.clip(
            dv_target,
            -self.a_brake * elapsed[None, :],
            self.a_accel * elapsed[None, :],
        )
        v_prof = np.maximum(v0 + dv, 0.0)  # 不允许倒车
        in_delay = times[None, :] < self.tau_delay
        d_prof = np.where(in_delay, d0, d_cmds[:, None])
        return v_prof, d_prof

    def _evaluate_candidates(self, v_cmds, d_cmds, points, r_eff, n_steps):
        """向量化评估候选集合：Ackermann 前推 + 三圆净空。

        返回 (c_min (n_cand,), poses (n_cand, n_steps, 3))。
        c_min = 预测轨迹上所有位姿、三个包络圆到所有障碍点的最小距离 - r_eff。
        无障碍点时 c_min 取 roi_radius（视为完全空旷）。
        """
        v_cmds = np.asarray(v_cmds, dtype=np.float64)
        d_cmds = np.clip(np.asarray(d_cmds, dtype=np.float64),
                         -self.max_steer, self.max_steer)
        n_cand = v_cmds.size
        dt = self.prediction_dt

        v_prof, d_prof = self._control_profiles(v_cmds, d_cmds, n_steps)

        # 逐步积分自行车模型（步数 <= 18，循环开销可忽略；候选维度向量化）
        x = np.zeros(n_cand)
        y = np.zeros(n_cand)
        psi = np.zeros(n_cand)
        poses = np.empty((n_cand, n_steps, 3), dtype=np.float64)
        tan_L = np.tan(d_prof) / self.wheelbase
        for k in range(n_steps):
            vk = v_prof[:, k]
            x = x + vk * np.cos(psi) * dt
            y = y + vk * np.sin(psi) * dt
            psi = psi + vk * tan_L[:, k] * dt
            poses[:, k, 0] = x
            poses[:, k, 1] = y
            poses[:, k, 2] = psi

        if points.shape[0] == 0:
            return np.full(n_cand, self.roi_radius), poses

        # 三圆圆心：(n_cand, n_steps, 3)
        cos_psi = np.cos(poses[:, :, 2])
        sin_psi = np.sin(poses[:, :, 2])
        cx = poses[:, :, 0:1] + self.circle_offsets[None, None, :] * cos_psi[:, :, None]
        cy = poses[:, :, 1:2] + self.circle_offsets[None, None, :] * sin_psi[:, :, None]

        # 到所有障碍点的最小平方距离：(n_cand, n_steps, 3, n_pts) → min
        dx = cx[:, :, :, None] - points[None, None, None, :, 0]
        dy = cy[:, :, :, None] - points[None, None, None, :, 1]
        d2 = dx * dx + dy * dy
        min_d = np.sqrt(d2.min(axis=(1, 2, 3)))
        return min_d - r_eff, poses

    def _pose_clearance(self, points, pose=(0.0, 0.0, 0.0)):
        """单一位姿的观测净空：三圆到障碍点的最小距离 - 几何半径（不含余量）。"""
        if points.shape[0] == 0:
            return float(self.roi_radius)
        x, y, psi = pose
        cx = x + self.circle_offsets * math.cos(psi)
        cy = y + self.circle_offsets * math.sin(psi)
        dx = cx[:, None] - points[None, :, 0]
        dy = cy[:, None] - points[None, :, 1]
        min_d = math.sqrt(float((dx * dx + dy * dy).min()))
        return min_d - self.circle_radius_geom

    def _update_ttc(self, r_raw, angle_min, angle_increment,
                    range_min, range_max, stamp_sec, direction):
        """连续两帧径向 closing speed 的保守型方向性 TTC。

        对未来行进方向 ±ttc_sector_half_width 扇区内、正在接近的 beam 计算
        TTC_i = max(r_i - 车头前伸距, 0.01) / closing_i，取低分位数（抗噪声）。
        该 TTC 含自车运动分量，属于保守估计。返回后同时缓存本帧扫描。
        """
        ttc = float("inf")
        n = r_raw.size
        lower = max(float(range_min), 0.05)
        upper = float(range_max) * 0.999 if range_max > 0 else np.inf
        finite = np.isfinite(r_raw)
        valid_now = finite & (r_raw > lower) & (r_raw < upper)

        if (
            self._prev_ranges is not None
            and self._prev_ranges.size == n
            and self._prev_stamp is not None
        ):
            dt = stamp_sec - self._prev_stamp
            if 1e-3 < dt < 0.5:
                angles = angle_min + np.arange(n, dtype=np.float64) * angle_increment
                # 扇区角度差归一化到 [-pi, pi]
                diff = np.arctan2(np.sin(angles - direction), np.cos(angles - direction))
                sector = np.abs(diff) <= self.ttc_sector_half_width
                both = valid_now & self._prev_valid & sector
                if np.any(both):
                    closing = (self._prev_ranges[both] - r_raw[both]) / dt
                    approaching = closing > self.ttc_min_closing
                    if np.count_nonzero(approaching) >= self.ttc_min_points:
                        r_eff = np.maximum(
                            r_raw[both][approaching] - self.front_reach_from_laser, 0.01
                        )
                        ttc_all = r_eff / closing[approaching]
                        ttc = float(np.percentile(ttc_all, self.ttc_percentile))

        # 缓存本帧供下一帧使用
        self._prev_ranges = r_raw.copy()
        self._prev_valid = valid_now
        return ttc

    def _finish_frame(self, executed_speed, executed_steer,
                      pred_pose, points, clearance_now, stamp_sec):
        """帧末状态更新：保形残差入窗、生成对下一帧净空的预测、记录已执行指令。"""
        # 1) 上一帧的预测 vs 本帧实际：只累计"预测过于乐观"的危险侧残差
        #    e_t = max(0, c_hat_{t|t-1} - c_t)
        if self._pred_next_clearance is not None and math.isfinite(clearance_now):
            score = max(0.0, self._pred_next_clearance - clearance_now)
            self._scores.append(score)

        # 2) 用"实际执行"的轨迹在 +frame_dt 处的位姿，对本帧点云预测下一帧净空
        if points.shape[0] > 0:
            self._pred_next_clearance = self._pose_clearance(points, pred_pose)
        else:
            self._pred_next_clearance = None

        # 3) 延迟建模用的"当前指令"估计与帧计数
        self._last_cmd_speed = float(executed_speed)
        self._last_cmd_steer = float(executed_steer)
        self._prev_stamp = float(stamp_sec)
        self.frames_processed += 1

    def _make_debug(self, level, reason, base_speed, base_steering,
                    suggested_speed, suggested_steering, c_min, ttc,
                    q_t, r_eff, n_points, t_start, degraded):
        applied = (not self.shadow_mode) and (
            abs(suggested_speed - base_speed) > 1e-9
            or abs(suggested_steering - base_steering) > 1e-9
        )
        return {
            "intervention_level": int(level),
            "intervention_reason": reason,
            "base_speed": float(base_speed),
            "base_steering": float(base_steering),
            "suggested_speed": float(suggested_speed),
            "suggested_steering": float(suggested_steering),
            "applied": bool(applied),
            "predicted_clearance": float(c_min),
            "ttc": float(ttc),
            "conformal_margin": float(q_t),
            "effective_radius": float(r_eff),
            "compute_time_ms": (time.perf_counter() - t_start) * 1000.0,
            "num_points": int(n_points),
            "sensor_degraded": bool(degraded),
            "shadow_mode": bool(self.shadow_mode),
        }
