"""反应式控制律（FTG + 走廊居中 + PD + 曲率限速），纯 numpy、雷达几何无关。

这是 doc§1.1 描述的「手写版对位件」的干净复现：
  - 方向项 term1：朝最优缝隙转（引力），前方开阔时衰减转向（下限 0.7）
  - 居中项 term2：左右距离不对称度（斥力居中）
  - PD 转向：delta = P*angle + D*(angle - angle_prev)
  - 限速律：v = base*(wd*exp(-|angle|)+wb)，转角越大越慢

用途：
  1) 部署兜底（student_policy_node 在学生超时/低置信时切到它）
  2) 反应式基线（§5「控制后端」消融的 A 对照；也可作专家候选）

注意：车上正式反应式仍是 roboracer_china_2025/battle_fast2_node.py（不改动）；
本模块只为「仿真 + 兜底 + 消融」提供一个几何无关、可复现的等价件。
"""
import math
import numpy as np


class ReactiveController:
    def __init__(self,
                 fov_rad: float = math.radians(180.0),  # 只看前向扇区
                 max_range: float = 10.0,
                 dir_detect_thresh: float = 2.5,         # 方向探测饱和距离 Dth
                 bubble_rad_m: float = 0.30,             # 最近障碍安全气泡
                 p_gain: float = 1.1,
                 d_gain: float = 0.2,
                 w_center: float = 0.3,
                 steer_limit: float = 0.34,
                 speed_base: float = 1.5,
                 speed_min: float = 0.4,
                 speed_max: float = 2.0):
        self.fov = fov_rad
        self.max_range = max_range
        self.dth = dir_detect_thresh
        self.bubble = bubble_rad_m
        self.P = p_gain
        self.D = d_gain
        self.w_center = w_center
        self.steer_limit = steer_limit
        self.speed_base = speed_base
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.last_angle = 0.0

    def reset(self):
        self.last_angle = 0.0

    def _preprocess(self, ranges, angles):
        r = np.asarray(ranges, dtype=np.float64).copy()
        r[~np.isfinite(r)] = self.max_range
        r = np.clip(r, 0.0, self.max_range)
        # 只保留前向扇区
        mask = np.abs(angles) <= (self.fov / 2.0)
        return r, mask

    def compute(self, ranges, angle_min: float, angle_increment: float):
        """输入一帧 scan，返回 (steering_rad, speed_mps)。"""
        ranges = np.asarray(ranges, dtype=np.float64)
        n = ranges.shape[0]
        angles = angle_min + np.arange(n) * angle_increment
        r, mask = self._preprocess(ranges, angles)
        idx = np.where(mask)[0]
        if idx.size < 5:
            return 0.0, self.speed_min

        rf = r[idx]
        af = angles[idx]

        # --- FTG 气泡：把最近障碍周围置零 ---
        proc = rf.copy()
        nearest = int(np.argmin(proc))
        if angle_increment > 1e-9 and proc[nearest] > 1e-3:
            half = int(max(1, math.ceil((self.bubble / max(proc[nearest], 1e-3)) / angle_increment)))
            lo = max(0, nearest - half)
            hi = min(proc.size, nearest + half + 1)
            proc[lo:hi] = 0.0

        # --- 找最大连续缝隙，取缝隙内最深点为目标 ---
        free = proc > 1e-3
        best_lo, best_hi, cur_lo = 0, 0, None
        for i in range(proc.size):
            if free[i] and cur_lo is None:
                cur_lo = i
            if (not free[i] or i == proc.size - 1) and cur_lo is not None:
                hi = i if not free[i] else i + 1
                if hi - cur_lo > best_hi - best_lo:
                    best_lo, best_hi = cur_lo, hi
                cur_lo = None
        if best_hi <= best_lo:
            target_local = nearest
            max_dis = float(rf[nearest])
        else:
            seg = proc[best_lo:best_hi]
            target_local = best_lo + int(np.argmax(seg))
            max_dis = float(rf[target_local])

        target_angle = float(af[target_local])

        # --- term1 方向项（朝缝隙；前方开阔衰减）---
        decay = max(math.exp(-max_dis / max(self.dth, 1e-6)), 0.7)
        term1 = -decay * (-target_angle)   # target_angle>0=左；转向左为正
        # --- term2 居中项（左右不对称度）---
        d_left = float(np.mean(rf[-max(1, rf.size // 10):]))
        d_right = float(np.mean(rf[:max(1, rf.size // 10)]))
        denom = d_left + d_right
        term2 = (d_left - d_right) / denom if denom > 1e-6 else 0.0

        angle = term1 + self.w_center * term2
        steer = self.P * angle + self.D * (angle - self.last_angle)
        self.last_angle = angle
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))

        # --- 限速律 ---
        speed = self.speed_base * (0.3 * math.exp(-np.clip(abs(angle), 0.0, 0.5)) + 0.7)
        speed = float(np.clip(speed, self.speed_min, self.speed_max))
        return steer, speed
