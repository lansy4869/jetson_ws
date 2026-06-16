"""W7：DAgger 迭代——修正 BC 的协变量漂移，误差 O(eps*T^2) -> O(eps*T)，仍不需奖励。

流程：滚学生 → 在其访问状态上查询专家动作 → 聚合 → 重训（从聚合集重新拟合）。
第 i 轮专家接管概率 beta = beta_decay**i（学生逐渐主导执行，专家始终提供标签）。

用法：
  python3 -m slash_distill.dagger --init-data data/demos.npz --init-ckpt ckpt/student_best.pt
  # smoke-test：
  python3 -m slash_distill.dagger --init-data /tmp/demo_smoke.npz --iters 1 \
      --rollouts 2 --max-steps 50 --epochs 2 --out /tmp/ckpt_dagger
"""
import argparse
import os
import numpy as np

from .params import GroundTruthLimits, load_distill
from .sim.rollout import make_env_and_expert, collect_episode
from . import train_bc


def main(argv=None):
    ap = argparse.ArgumentParser(description="W7 DAgger 迭代")
    ap.add_argument("--init-data", required=True, help="W5 采的 BC 数据集 .npz（聚合起点）")
    ap.add_argument("--init-ckpt", default=None, help="BC 初始 ckpt；缺省则先在 init-data 上训一个")
    ap.add_argument("--config", default="distill.yaml")
    ap.add_argument("--maps", nargs="*", default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--rollouts", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="每轮重训 epochs")
    ap.add_argument("--num-agents", type=int, default=1)
    ap.add_argument("--out", default="ckpt_dagger")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from .models.policy import TorchStudentPolicy

    limits = GroundTruthLimits.from_yaml()
    dcfg = load_distill(args.config)
    dc = dcfg.get("dagger", {})
    iters = args.iters if args.iters is not None else int(dc.get("iters", 5))
    rollouts = args.rollouts if args.rollouts is not None else int(dc.get("rollouts_per_iter", 20))
    epochs = args.epochs if args.epochs is not None else int(dc.get("epochs_per_iter", 30))
    max_steps = args.max_steps if args.max_steps is not None else int(dcfg.get("collect", {}).get("max_steps", 2000))
    beta_decay = float(dc.get("beta_decay", 0.7))
    maps = args.maps or (dcfg.get("collect", {}).get("maps") or ["vegas"])
    if len(maps) == 0:
        maps = ["vegas"]
    os.makedirs(args.out, exist_ok=True)

    # 聚合数据起点
    d0 = np.load(args.init_data)
    agg_l = [d0["lidar"].astype(np.float32)]
    agg_s = [d0["scalar"].astype(np.float32)]
    agg_a = [d0["action"].astype(np.float32)]
    meta = dict(n_layers=int(d0["n_layers"]), n_beams=int(d0["n_beams"]),
                scalar_dim=int(d0["scalar_dim"]),
                steer_max=float(d0["steer_max"]), v_max=float(d0["v_max"]))
    agg_path = os.path.join(args.out, "aggregated.npz")

    def save_agg():
        np.savez_compressed(agg_path,
                            lidar=np.concatenate(agg_l, 0),
                            scalar=np.concatenate(agg_s, 0),
                            action=np.concatenate(agg_a, 0),
                            **meta)

    # 初始 ckpt
    ckpt = args.init_ckpt
    if ckpt is None or not os.path.isfile(ckpt):
        print("[dagger] 无 init-ckpt，先在 init-data 上训一个 BC 学生")
        save_agg()
        bc_argv = ["--data", agg_path, "--config", args.config,
                   "--epochs", str(epochs), "--out", args.out]
        if args.device:
            bc_argv += ["--device", args.device]
        train_bc.main(bc_argv)
        ckpt = os.path.join(args.out, "student_best.pt")

    rng = np.random.default_rng(args.seed)
    for it in range(1, iters + 1):
        beta = beta_decay ** it
        policy = TorchStudentPolicy(ckpt, device=(args.device or "cpu"))
        n_new = 0; coll = 0
        for r in range(rollouts):
            mp = maps[r % len(maps)]
            env, expert, opp, obs_cfg = make_env_and_expert(
                limits, dcfg, mp, grounding="sysid", seed=args.seed + it * 1000 + r,
                num_agents=args.num_agents, domain_rand=dcfg.get("collect", {}).get("domain_rand", {}))
            try:
                data, m = collect_episode(env, expert, opp, obs_cfg, max_steps,
                                          student_act=policy.act, beta=beta, rng=rng)
            finally:
                env.close()
            agg_l.append(data["lidar"]); agg_s.append(data["scalar"]); agg_a.append(data["action"])
            n_new += len(data["lidar"]); coll += int(m["collision"])
        save_agg()
        total = sum(len(x) for x in agg_l)
        print(f"[dagger] iter {it}/{iters} beta={beta:.3f} new_frames={n_new} "
              f"agg_total={total} rollout_collisions={coll}/{rollouts} -> 重训")
        bc_argv = ["--data", agg_path, "--config", args.config,
                   "--epochs", str(epochs), "--out", args.out]
        if args.device:
            bc_argv += ["--device", args.device]
        train_bc.main(bc_argv)
        ckpt = os.path.join(args.out, "student_best.pt")

    print(f"[dagger] 完成，最终 ckpt -> {ckpt}")


if __name__ == "__main__":
    main()
