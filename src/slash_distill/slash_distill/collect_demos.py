"""W5：在 sysid 接地仿真里跑特权专家，采集 (obs_student, action_expert) 数据集。

用法：
  python3 -m slash_distill.collect_demos --episodes 200 --out data/demos.npz
  python3 -m slash_distill.collect_demos --maps vegas berlin --num-agents 2
  # smoke-test：
  python3 -m slash_distill.collect_demos --episodes 2 --max-steps 50 --out /tmp/demo_smoke.npz

输出 .npz：lidar(N,C,B) scalar(N,S) action(N,2=steer,speed) + meta(n_layers,n_beams,scalar_dim,steer_max,v_max)
"""
import argparse
import os
import numpy as np

from .params import GroundTruthLimits, load_distill
from .sim.rollout import make_env_and_expert, collect_episode


def main(argv=None):
    ap = argparse.ArgumentParser(description="W5 专家示范采集")
    ap.add_argument("--config", default="distill.yaml")
    ap.add_argument("--maps", nargs="*", default=None, help="地图名/路径；默认用 config 或 vegas")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--num-agents", type=int, default=None)
    ap.add_argument("--grounding", default="sysid", choices=["sysid", "default"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args(argv)

    limits = GroundTruthLimits.from_yaml()
    dcfg = load_distill(args.config)
    cc = dcfg.get("collect", {})

    maps = args.maps or (cc.get("maps") or ["vegas"])
    if len(maps) == 0:
        maps = ["vegas"]
    episodes = args.episodes if args.episodes is not None else int(cc.get("n_episodes", 200))
    max_steps = args.max_steps if args.max_steps is not None else int(cc.get("max_steps", 2000))
    num_agents = args.num_agents if args.num_agents is not None else int(cc.get("num_agents", 1))
    out = args.out or cc.get("out", "data/demos.npz")
    dr = cc.get("domain_rand", {})
    jitter = float(dr.get("start_jitter_m", 0.0)) if dr.get("enable", False) else 0.0

    rng = np.random.default_rng(args.seed)
    all_l, all_s, all_a = [], [], []
    tot_collisions = 0
    for ep in range(episodes):
        mp = maps[ep % len(maps)]
        seed = args.seed + ep
        env, expert, opp, obs_cfg = make_env_and_expert(
            limits, dcfg, mp, grounding=args.grounding, seed=seed,
            num_agents=num_agents, domain_rand=dr)
        try:
            data, m = collect_episode(env, expert, opp, obs_cfg, max_steps,
                                      start_jitter_m=jitter, rng=rng)
        finally:
            env.close()
        all_l.append(data["lidar"]); all_s.append(data["scalar"]); all_a.append(data["action"])
        tot_collisions += int(m["collision"])
        print(f"[collect] ep {ep+1}/{episodes} map={mp} steps={m['steps']} "
              f"collision={m['collision']} laps={m['lap_count']:.0f} "
              f"frames={len(data['lidar'])} expert={'pursuit' if expert.uses_path else 'reactive'}")

    lidar = np.concatenate(all_l, axis=0) if all_l else np.zeros((0,))
    scalar = np.concatenate(all_s, axis=0) if all_s else np.zeros((0,))
    action = np.concatenate(all_a, axis=0) if all_a else np.zeros((0,))

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(
        out, lidar=lidar, scalar=scalar, action=action,
        n_layers=obs_cfg.n_layers, n_beams=obs_cfg.n_beams, scalar_dim=obs_cfg.scalar_dim,
        steer_max=dcfg.get("action", {}).get("steer_max", limits.steering_limit_rad),
        v_max=dcfg.get("action", {}).get("v_max", limits.v_max))
    print(f"[collect] 保存 {action.shape[0]} 帧 -> {out} "
          f"(lidar {lidar.shape}, scalar {scalar.shape}); 总碰撞 episode {tot_collisions}/{episodes}")


if __name__ == "__main__":
    main()
