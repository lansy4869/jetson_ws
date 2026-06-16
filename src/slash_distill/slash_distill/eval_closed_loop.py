"""W6/W8：闭环评估学生（或纯反应式）在 sysid 接地仿真里的表现。

链路：obs → 学生(可选) → [低置信/异常 → 反应式兜底] → [g-g-v 护盾] → env.step
指标：完成率(无碰撞)、碰撞率、平均圈数、平均速度、推理时延、回退触发率、护盾夹紧率。

用法：
  python3 -m slash_distill.eval_closed_loop --ckpt ckpt/student_best.pt --episodes 10
  python3 -m slash_distill.eval_closed_loop --reactive            # 纯反应式基线
  python3 -m slash_distill.eval_closed_loop --ckpt ... --no-shield --no-fallback
  # smoke-test：
  python3 -m slash_distill.eval_closed_loop --reactive --episodes 1 --max-steps 60
"""
import argparse
import time
import numpy as np

from .params import GroundTruthLimits, load_distill
from .common import obs as obsmod
from .common.ggv import GgvProjector
from .common.reactive import ReactiveController
from .sim.grounded_env import GroundedF110
from .sim.opponent import ReactiveOpponent
from .experts.centerline import load_waypoints, find_centerline_csv


def evaluate(policy, limits, dcfg, map_name="vegas", grounding="sysid",
             n_episodes=10, max_steps=3000, use_shield=True, use_fallback=True,
             num_agents=1, seed=0, render=False):
    """policy: 对象含 act(lidar,scalar)->(steer,speed)；None=纯反应式。返回 metrics dict。"""
    ec = dcfg.get("eval", {})
    fb_jump = float(ec.get("fallback_speed_jump", 1.5))
    obs_cfg = obsmod.ObsConfig.from_distill(dcfg, n_layers=dcfg.get("obs", {}).get("n_layers_sim", 1))
    ggv = GgvProjector.from_limits(limits)

    coll = 0; laps = []; steps_list = []; speeds = []
    inf_ms = []; fb_trig = 0; fb_tot = 0; clip = 0; clip_tot = 0

    for ep in range(n_episodes):
        env = GroundedF110(limits=limits, map_name=map_name, num_agents=num_agents,
                           grounding=grounding, seed=seed + ep)
        reactive = ReactiveController(steer_limit=limits.steering_limit_rad, speed_max=limits.v_max)
        opp = ReactiveOpponent(env.angle_min, env.angle_increment,
                               speed_max=0.6 * limits.v_max,
                               steer_limit=limits.steering_limit_rad) if num_agents >= 2 else None
        ob = env.reset()
        ep_speed = []
        try:
            for st in range(1, max_steps + 1):
                lidar, scalar = obsmod.build_obs(
                    [(ob['scan'], ob['angle_min'], ob['angle_increment'])],
                    ob['ego_speed'], ob['opponent_state'], obs_cfg)

                t0 = time.perf_counter()
                if policy is None:
                    steer, speed = reactive.compute(ob['scan'], ob['angle_min'], ob['angle_increment'])
                else:
                    steer, speed = policy.act(lidar, scalar)
                inf_ms.append((time.perf_counter() - t0) * 1e3)

                # 反应式兜底
                if use_fallback and policy is not None:
                    fb_tot += 1
                    r_steer, r_speed = reactive.compute(ob['scan'], ob['angle_min'], ob['angle_increment'])
                    bad = (not np.isfinite(steer)) or (not np.isfinite(speed)) or \
                          (abs(speed - r_speed) > fb_jump)
                    if bad:
                        steer, speed = r_steer, r_speed
                        fb_trig += 1

                # g-g-v 护盾
                if use_shield:
                    clip_tot += 1
                    s2, v2, _ = ggv.project(steer, speed, ob['ego_speed'], env.control_dt)
                    if abs(s2 - steer) > 1e-3 or abs(v2 - speed) > 1e-2:
                        clip += 1
                    steer, speed = s2, v2

                opp_action = opp.act(ob) if opp is not None else None
                ob = env.step((steer, speed), opp_action)
                ep_speed.append(ob['ego_speed'])
                if render:
                    env.render()
                if ob['collision']:
                    coll += 1
                    break
                if ob['done']:
                    break
        finally:
            env.close()
        laps.append(ob['lap_count']); steps_list.append(st)
        speeds.append(float(np.mean(ep_speed)) if ep_speed else 0.0)

    m = {
        "episodes": n_episodes,
        "collision_rate": coll / max(n_episodes, 1),
        "completion_rate": 1.0 - coll / max(n_episodes, 1),
        "mean_lap_count": float(np.mean(laps)) if laps else 0.0,
        "mean_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "mean_inference_ms": float(np.mean(inf_ms)) if inf_ms else 0.0,
        "fallback_rate": fb_trig / max(fb_tot, 1),
        "shield_clip_rate": clip / max(clip_tot, 1),
    }
    return m


def _print_metrics(tag, m):
    print(f"[eval:{tag}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                      for k, v in m.items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description="W6/W8 闭环评估")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--reactive", action="store_true", help="纯反应式基线（不加载学生）")
    ap.add_argument("--config", default="distill.yaml")
    ap.add_argument("--map", default="vegas")
    ap.add_argument("--grounding", default="sysid", choices=["sysid", "default"])
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--num-agents", type=int, default=1)
    ap.add_argument("--no-shield", action="store_true")
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args(argv)

    limits = GroundTruthLimits.from_yaml()
    dcfg = load_distill(args.config)
    ec = dcfg.get("eval", {})
    n_ep = args.episodes if args.episodes is not None else int(ec.get("n_episodes", 10))
    max_steps = args.max_steps if args.max_steps is not None else int(ec.get("max_steps", 3000))

    policy = None
    if not args.reactive:
        if not args.ckpt:
            ap.error("需 --ckpt 或 --reactive")
        from .models.policy import TorchStudentPolicy
        policy = TorchStudentPolicy(args.ckpt, device=args.device)

    m = evaluate(policy, limits, dcfg, map_name=args.map, grounding=args.grounding,
                 n_episodes=n_ep, max_steps=max_steps,
                 use_shield=not args.no_shield, use_fallback=not args.no_fallback,
                 num_agents=args.num_agents, seed=args.seed, render=args.render)
    _print_metrics("reactive" if args.reactive else "student", m)


if __name__ == "__main__":
    main()
