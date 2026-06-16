"""仿真对手车策略（num_agents=2 时驱动第二 agent）。

用反应式控制律驱动对手，速度上限可调（让对手比 ego 慢，便于产生超车/跟随示范）。
对手的 scan 从 env 的第二 agent 取（grounded_env 目前只暴露 ego scan，这里直接吃 raw obs）。
"""
import numpy as np
from ..common.reactive import ReactiveController


class ReactiveOpponent:
    def __init__(self, angle_min: float, angle_increment: float,
                 speed_max: float = 1.2, steer_limit: float = 0.34):
        self.angle_min = angle_min
        self.angle_increment = angle_increment
        self.ctrl = ReactiveController(steer_limit=steer_limit, speed_max=speed_max)

    def reset(self):
        self.ctrl.reset()

    def act(self, raw_obs: dict):
        """raw_obs = grounded_env wrapped obs（含 'raw' = gym obs）。"""
        gym_obs = raw_obs['raw']
        if len(gym_obs['scans']) < 2:
            return 0.0, 0.0
        scan = np.asarray(gym_obs['scans'][1], dtype=np.float64)
        return self.ctrl.compute(scan, self.angle_min, self.angle_increment)
