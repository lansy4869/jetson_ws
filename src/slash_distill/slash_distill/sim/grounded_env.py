"""sysid 接地的 f110_gym 包装（W5 创新点3 的「接地仿真」）。

接地内容：
  - 几何：lf+lr = 0.25m（替换 gym 默认 0.33m，正是文档要打的 sim-to-real 失配点）
  - 极限：s_min/s_max=±0.34, a_max=ax_accel, v_max, mu≈ay_max/g
  - 命令侧：稳态增益 K（指令速度×K）+ 一阶滞后 tau + 纯延迟 Td（逼近 sysid 纵向模型）
  - 决策频率：control_hz（gym 物理 dt=0.01，每 round(1/(hz*dt)) 个物理步出 1 个动作）
  - 域随机化：mu / lidar 噪声 / 起点扰动

grounding='default' 时使用 gym 文献默认参数（0.33m/未标定）→ 供「sysid vs 默认」消融对照。

统一 obs（dict）：
  scan, angle_min, angle_increment, ego_speed, ego_pose(x,y,theta), ego_yawrate,
  collision(bool), lap_count, lap_time, opponent_state[10]（num_agents=2 时由 poses 推出，否则全零）
"""
import math
import numpy as np

from ..params import GroundTruthLimits

G = 9.81


def _gym():
    import gym  # 延迟导入，避免无 gym 环境下 import 包就炸
    return gym


def build_params(limits: GroundTruthLimits, grounding: str = "sysid") -> dict:
    """构造 gym vehicle params。sysid: 接地；default: 文献默认。"""
    base = {
        'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562,
        'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712,
        's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2,
        'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0,
        'width': 0.31, 'length': 0.58,
    }
    if grounding == "default":
        base['lf'] = limits.gym_defaults.get('lf', base['lf'])
        base['lr'] = limits.gym_defaults.get('lr', base['lr'])
        base['mu'] = limits.gym_defaults.get('mu', base['mu'])
        return base
    # --- sysid 接地 ---
    half = limits.wheelbase / 2.0
    base['lf'] = half
    base['lr'] = half
    base['s_min'] = -limits.steering_limit_rad
    base['s_max'] = limits.steering_limit_rad
    base['a_max'] = limits.ax_accel_max
    base['v_min'] = -0.1
    base['v_max'] = max(limits.v_max, 1.0)
    base['mu'] = max(0.1, limits.ay_max / G)   # 横向极限接地（ay_max 为假设时此项也是假设）
    return base


class GroundedF110:
    def __init__(self,
                 limits: GroundTruthLimits = None,
                 map_name: str = "vegas",
                 map_ext: str = ".png",
                 num_agents: int = 1,
                 grounding: str = "sysid",
                 control_hz: float = 50.0,
                 physics_dt: float = 0.01,
                 seed: int = 0,
                 lidar_noise_std: float = 0.0,
                 mu_override: float = None):
        self.limits = limits or GroundTruthLimits()
        self.num_agents = int(num_agents)
        self.grounding = grounding
        self.control_dt = 1.0 / float(control_hz)
        self.physics_dt = float(physics_dt)
        self.substeps = max(1, int(round(self.control_dt / self.physics_dt)))
        self.seed = int(seed)
        self.lidar_noise_std = float(lidar_noise_std)

        self.params = build_params(self.limits, grounding)
        if mu_override is not None:
            self.params['mu'] = float(mu_override)

        gym = _gym()
        self.env = gym.make('f110_gym:f110-v0',
                            map=self._resolve_map(map_name),
                            map_ext=self._resolve_ext(map_name, map_ext),
                            num_agents=self.num_agents,
                            timestep=self.physics_dt,
                            seed=self.seed,
                            params=self.params)
        # 几何
        self.num_beams = 1080
        self.fov = 4.7
        self.angle_min = -self.fov / 2.0
        self.angle_increment = self.fov / (self.num_beams - 1)

        # 命令侧接地状态
        self.K = self.limits.long_gain_K if grounding == "sysid" else 1.0
        self.tau = self.limits.long_tau_s if grounding == "sysid" else 0.0
        self.delay_steps = int(round((self.limits.long_delay_s if grounding == "sysid" else 0.0)
                                     / self.physics_dt))
        self._filt_speed = [0.0] * self.num_agents
        self._delay_buf = [[] for _ in range(self.num_agents)]

        self._last_obs = None
        self._prev_opp_xy = None
        self._rng = np.random.default_rng(self.seed)

    # ----------------------------------------------------- 地图
    @staticmethod
    def _resolve_map(map_name: str) -> str:
        """builtin 名 → gym maps 目录的绝对路径前缀（gym 会自动补 .yaml）。
        自定义路径 → 去掉 .yaml 后缀原样返回。"""
        import os
        builtin = {"vegas", "berlin", "skirk", "levine", "stata_basement"}
        if map_name in builtin:
            try:
                import f110_gym
                return os.path.join(os.path.dirname(f110_gym.__file__), "envs", "maps", map_name)
            except Exception:
                return map_name
        if map_name.endswith(".yaml"):
            map_name = map_name[:-5]
        return map_name

    @staticmethod
    def _resolve_ext(map_name: str, map_ext: str) -> str:
        # levine 自带为 .pgm，其余 builtin 为 .png
        if map_name == "levine":
            return ".pgm"
        return map_ext

    # ----------------------------------------------------- reset / step
    def reset(self, poses=None, start_jitter_m: float = 0.0):
        if poses is None:
            poses = np.zeros((self.num_agents, 3), dtype=np.float64)
            if self.num_agents >= 2:
                poses[1, 0] = 1.5   # 对手在前方 1.5m
        poses = np.asarray(poses, dtype=np.float64).reshape(self.num_agents, 3)
        if start_jitter_m > 0:
            poses[:, :2] += self._rng.normal(0.0, start_jitter_m, size=(self.num_agents, 2))
        self._filt_speed = [0.0] * self.num_agents
        self._delay_buf = [[] for _ in range(self.num_agents)]
        self._prev_opp_xy = None
        obs, _, done, info = self.env.reset(poses)
        self._last_obs = obs
        return self._wrap(obs, done, info)

    def _ground_speed_cmd(self, agent_i: int, v_cmd: float) -> float:
        """命令侧：K 增益 + 一阶滞后 + 纯延迟。"""
        v = self.K * float(v_cmd)
        # 纯延迟
        buf = self._delay_buf[agent_i]
        if self.delay_steps > 0:
            buf.append(v)
            v = buf.pop(0) if len(buf) > self.delay_steps else (buf[0] if buf else v)
        # 一阶滞后（在 substep 累积，这里用 control_dt 近似）
        if self.tau > 1e-6:
            alpha = self.control_dt / (self.tau + self.control_dt)
            self._filt_speed[agent_i] += alpha * (v - self._filt_speed[agent_i])
            v = self._filt_speed[agent_i]
        return v

    def step(self, ego_action, opp_action=None):
        """ego_action=(steer, speed)；opp_action 同（num_agents>=2 时）。"""
        actions = np.zeros((self.num_agents, 2), dtype=np.float64)
        es, ev = float(ego_action[0]), float(ego_action[1])
        actions[0, 0] = es
        actions[0, 1] = self._ground_speed_cmd(0, ev)
        if self.num_agents >= 2 and opp_action is not None:
            actions[1, 0] = float(opp_action[0])
            actions[1, 1] = self._ground_speed_cmd(1, float(opp_action[1]))

        done = False
        info = {}
        obs = self._last_obs
        for _ in range(self.substeps):
            obs, _, done, info = self.env.step(actions)
            if done:
                break
        self._last_obs = obs
        return self._wrap(obs, done, info)

    # ----------------------------------------------------- 观测打包
    def _scan(self, obs):
        s = np.asarray(obs['scans'][0], dtype=np.float64)
        if self.lidar_noise_std > 0:
            s = s + self._rng.normal(0.0, self.lidar_noise_std, size=s.shape)
            s = np.clip(s, 0.0, None)
        return s

    def _opponent_state(self, obs):
        """num_agents>=2：从 ego/opp poses 推出 W4 布局的 opponent_state[10]。"""
        out = np.zeros(10, dtype=np.float64)
        if self.num_agents < 2:
            return out
        ex, ey, eth = obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]
        ox, oy = obs['poses_x'][1], obs['poses_y'][1]
        dx, dy = ox - ex, oy - ey
        c, s = math.cos(-eth), math.sin(-eth)
        bx = c * dx - s * dy            # 车体系
        by = s * dx + c * dy
        rng = math.hypot(bx, by)
        bearing = math.atan2(by, bx)
        # 地速差（用上一帧对手 xy 数值微分，再扣自车？这里给地速近似）
        vx = vy = 0.0
        if self._prev_opp_xy is not None:
            pdx = (ox - self._prev_opp_xy[0]) / self.control_dt
            pdy = (oy - self._prev_opp_xy[1]) / self.control_dt
            vx = c * pdx - s * pdy
            vy = s * pdx + c * pdy
        self._prev_opp_xy = (ox, oy)
        speed = math.hypot(vx, vy)
        # range_rate：接近为负
        ego_v = float(obs['linear_vels_x'][0])
        range_rate = -(ego_v * math.cos(bearing)) + (vx * math.cos(bearing) + vy * math.sin(bearing))
        is_dyn = 1.0 if speed > 0.35 else 0.0
        out[:] = [1.0, rng, bearing, bx, by, vx, vy, speed, range_rate, is_dyn]
        return out

    def _wrap(self, obs, done, info):
        return {
            'scan': self._scan(obs),
            'angle_min': self.angle_min,
            'angle_increment': self.angle_increment,
            'ego_speed': float(obs['linear_vels_x'][0]),
            'ego_pose': (float(obs['poses_x'][0]), float(obs['poses_y'][0]), float(obs['poses_theta'][0])),
            'ego_yawrate': float(obs['ang_vels_z'][0]),
            'collision': bool(np.any(obs['collisions'])),
            'lap_count': float(obs['lap_counts'][0]),
            'lap_time': float(obs['lap_times'][0]),
            'opponent_state': self._opponent_state(obs),
            'done': bool(done),
            'raw': obs,
        }

    def render(self, mode="human_fast"):
        self.env.render(mode)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
