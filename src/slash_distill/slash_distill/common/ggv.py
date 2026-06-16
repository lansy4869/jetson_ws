"""g-g-v 安全壳（纯 Python，无 ROS 依赖）。

数学**逐行对齐** src/slash_safety/slash_safety/ggv_shield_node.py 的 `_shield`：
  1) 转向限幅：|delta| <= steering_limit_rad
  2) 过弯限速：kappa = tan(|delta|)/L,  v <= sqrt(ay_max*factor / kappa)
  3) 纵向变化率：dv 落在 [-ax_brake*dt, +ax_accel*dt]
  4) 全局夹紧：v in [v_min, v_max]

仿真 rollout 用本模块在每步投影学生/专家输出；实车部署仍用现成的 ggv_shield 节点
（学生节点只发 /drive_raw），二者数学一致，杜绝 sim-to-real 漂移。
"""
import math
from dataclasses import dataclass


@dataclass
class GgvProjector:
    wheelbase: float = 0.25
    steering_limit_rad: float = 0.34
    ax_accel_max: float = 6.35
    ax_brake_max: float = 6.66      # 正幅值
    ay_max: float = 9.81
    ay_safety_factor: float = 0.6
    v_max: float = 2.0
    v_min: float = 0.0

    @classmethod
    def from_limits(cls, lim) -> "GgvProjector":
        """从 params.GroundTruthLimits 构造。"""
        return cls(
            wheelbase=lim.wheelbase,
            steering_limit_rad=lim.steering_limit_rad,
            ax_accel_max=lim.ax_accel_max,
            ax_brake_max=abs(lim.ax_brake_max),
            ay_max=lim.ay_max,
            ay_safety_factor=lim.ay_safety_factor,
            v_max=lim.v_max,
            v_min=lim.v_min,
        )

    def curvature_speed_cap(self, steer: float) -> float:
        kappa = abs(math.tan(steer)) / max(self.wheelbase, 1e-6)
        if kappa < 1e-6:
            return self.v_max
        ay_eff = max(0.0, self.ay_max * self.ay_safety_factor)
        return min(self.v_max, math.sqrt(ay_eff / kappa))

    def rate_limit_speed(self, v_des: float, v_ref: float, dt: float) -> float:
        dv = v_des - v_ref
        dv = max(-abs(self.ax_brake_max) * dt, min(self.ax_accel_max * dt, dv))
        return v_ref + dv

    def project(self, steer_req: float, v_req: float, v_ref: float, dt: float):
        """把 (steer_req, v_req) 投影回可行域。返回 (steer, v, v_cap)。
        v_ref = 当前实测/上一步车速（纵向变化率参考）。"""
        steer = max(-self.steering_limit_rad, min(self.steering_limit_rad, float(steer_req)))
        v_cap = self.curvature_speed_cap(steer)
        v_des = max(self.v_min, min(v_cap, float(v_req)))
        v_out = self.rate_limit_speed(v_des, v_ref, dt)
        v_out = max(self.v_min, min(v_cap, v_out))
        return steer, v_out, v_cap
