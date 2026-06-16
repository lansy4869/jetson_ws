"""仿真 rollout 工具：构造 env+专家，采集带专家标签的数据（collect + DAgger 共用）。"""
import numpy as np

from ..params import GroundTruthLimits
from ..common import obs as obsmod
from ..experts.pursuit_expert import PursuitExpert
from ..experts.centerline import load_waypoints, find_centerline_csv
from .grounded_env import GroundedF110
from .opponent import ReactiveOpponent


def make_env_and_expert(limits: GroundTruthLimits, dcfg: dict, map_name: str,
                        grounding: str = "sysid", seed: int = 0,
                        num_agents: int = 1, domain_rand: dict = None):
    dr = domain_rand or {}
    rng = np.random.default_rng(seed)
    mu_override = None
    if dr.get("enable", False):
        lo, hi = dr.get("mu_range", [1.0, 1.0])
        mu_override = float(rng.uniform(lo, hi)) * (limits.ay_max / 9.81)
    lidar_noise = float(dr.get("lidar_noise_std", 0.0)) if dr.get("enable", False) else 0.0

    env = GroundedF110(limits=limits, map_name=map_name,
                       map_ext=dcfg.get("collect", {}).get("map_ext", ".png"),
                       num_agents=num_agents, grounding=grounding,
                       control_hz=dcfg.get("collect", {}).get("control_hz", 50),
                       seed=seed, lidar_noise_std=lidar_noise, mu_override=mu_override)

    wpts = load_waypoints(find_centerline_csv(map_name))
    expert = PursuitExpert(limits, waypoints=wpts)
    opp = ReactiveOpponent(env.angle_min, env.angle_increment,
                           speed_max=0.6 * limits.v_max,
                           steer_limit=limits.steering_limit_rad) if num_agents >= 2 else None
    obs_cfg = obsmod.ObsConfig.from_distill(dcfg, n_layers=dcfg.get("obs", {}).get("n_layers_sim", 1))
    return env, expert, opp, obs_cfg


def collect_episode(env: GroundedF110, expert: PursuitExpert, opp, obs_cfg,
                    max_steps: int, start_jitter_m: float = 0.0,
                    student_act=None, beta: float = 1.0, rng=None):
    """跑一个 episode，返回 (data, metrics)。

    data = dict(lidar=[...], scalar=[...], action=[...])，action = 专家原始 (steer,speed)。
    DAgger：student_act(lidar,scalar)->(steer,speed)；以概率 beta 用专家执行、否则用学生执行，
            但**标签始终是专家**（聚合学生访问状态上的专家动作）。
    """
    rng = rng or np.random.default_rng(0)
    expert.reset()
    if opp is not None:
        opp.reset()
    ob = env.reset(start_jitter_m=start_jitter_m)

    L, Sc, A = [], [], []
    collided = False
    steps = 0
    for steps in range(1, max_steps + 1):
        lidar, scalar = obsmod.build_obs(
            [(ob['scan'], ob['angle_min'], ob['angle_increment'])],
            ob['ego_speed'], ob['opponent_state'], obs_cfg)
        e_steer, e_speed = expert.act(ob)
        L.append(lidar); Sc.append(scalar); A.append([e_steer, e_speed])

        if student_act is not None and rng.random() > beta:
            s_steer, s_speed = student_act(lidar, scalar)
            exec_action = (s_steer, s_speed)
        else:
            exec_action = (e_steer, e_speed)

        opp_action = opp.act(ob) if opp is not None else None
        ob = env.step(exec_action, opp_action)
        if ob['collision']:
            collided = True
            break
        if ob['done']:
            break

    metrics = {"steps": steps, "collision": collided,
               "lap_count": ob['lap_count'], "lap_time": ob['lap_time']}
    data = {"lidar": np.asarray(L, dtype=np.float32),
            "scalar": np.asarray(Sc, dtype=np.float32),
            "action": np.asarray(A, dtype=np.float32)}
    return data, metrics
