"""特权专家：中心线 pure-pursuit + g-g-v 限速 profile（W5）。

"特权"含义：训练时可用、部署丢弃的信息——
  - 地图中心线 waypoints（全局路径）
  - 自车全状态真值（pose/速度）
  - 对手真值（opponent_state；用于 follow/避让减速）

控制律：
  - 转向：pure pursuit  delta = atan2(2 L sin(alpha), Ld)
  - 限速：v <= min(v_max, sqrt(ay_max*factor / kappa))，kappa 取「路径局部曲率」与「pursuit 弧曲率」较大者
  - 对手感知：前方近距对手 → 降速跟随（给 BC/DAgger 可学的避让行为）

无 waypoints（用户尚未放中心线）→ 退化为 ReactiveController（流程仍可跑通）。
"""
import math
import numpy as np

from ..common.ggv import GgvProjector
from ..common.reactive import ReactiveController
from .centerline import WaypointPath


class PursuitExpert:
    def __init__(self,
                 limits,
                 waypoints: np.ndarray = None,
                 lookahead: float = 1.0,
                 lookahead_gain: float = 0.3,
                 follow_range: float = 2.0,
                 follow_speed: float = 0.5):
        self.limits = limits
        self.L = limits.wheelbase
        self.ggv = GgvProjector.from_limits(limits)
        self.Ld0 = lookahead
        self.Ld_gain = lookahead_gain
        self.follow_range = follow_range
        self.follow_speed = follow_speed
        self.path = WaypointPath(waypoints) if waypoints is not None and len(waypoints) >= 3 else None
        self.reactive = ReactiveController(steer_limit=limits.steering_limit_rad,
                                           speed_max=limits.v_max)
        self.uses_path = self.path is not None
        self._prev_v = 0.0

    def reset(self):
        self.reactive.reset()
        self._prev_v = 0.0

    def act(self, obs: dict):
        """obs = grounded_env 的 wrapped obs。返回 (steer, speed)（未过护盾的专家原始指令）。"""
        if not self.uses_path:
            steer, speed = self.reactive.compute(
                obs['scan'], obs['angle_min'], obs['angle_increment'])
            speed = self._opponent_slowdown(obs, speed)
            self._prev_v = speed
            return steer, speed

        x, y, th = obs['ego_pose']
        v = max(0.0, obs['ego_speed'])
        Ld = self.Ld0 + self.Ld_gain * v
        target, ti = self.path.lookahead_point(x, y, Ld)

        # 目标点在车体系下的横向偏差 → pure pursuit
        dx, dy = target[0] - x, target[1] - y
        c, s = math.cos(-th), math.sin(-th)
        tx = c * dx - s * dy
        ty = s * dx + c * dy
        Ld_eff = max(math.hypot(tx, ty), 1e-3)
        alpha = math.atan2(ty, tx)
        steer = math.atan2(2.0 * self.L * math.sin(alpha), Ld_eff)
        steer = float(np.clip(steer, -self.limits.steering_limit_rad, self.limits.steering_limit_rad))

        # g-g-v 限速：路径曲率 vs pursuit 弧曲率取大
        kappa_path = self.path.curvature_at(ti)
        kappa_arc = abs(2.0 * math.sin(alpha) / Ld_eff)
        kappa = max(kappa_path, kappa_arc)
        wpt_v = self.path.v[ti]
        v_cap = self.ggv.curvature_speed_cap(steer)
        v_des = v_cap if math.isnan(wpt_v) else min(wpt_v, v_cap)
        v_des = min(v_des, self.limits.v_max)
        if kappa > 1e-6:
            v_des = min(v_des, math.sqrt(self.limits.ay_max * self.limits.ay_safety_factor / kappa))

        v_des = self._opponent_slowdown(obs, v_des)
        self._prev_v = v_des
        return steer, float(v_des)

    def _opponent_slowdown(self, obs: dict, v_des: float) -> float:
        """前方近距对手 → 降速跟随（特权：用对手真值）。"""
        opp = obs.get('opponent_state')
        if opp is None:
            return v_des
        valid, rng, bearing = float(opp[0]), float(opp[1]), float(opp[2])
        if valid > 0.5 and rng < self.follow_range and abs(bearing) < math.radians(30):
            return min(v_des, self.follow_speed)
        return v_des
