"""W8：跑 §5 消融矩阵（仿真可做的行）→ CSV。

可仿真的 4 个维度（感知维度=实车/定性，不在此）：
  vehicle_params : gym 默认(0.33m) vs sysid 接地        （用反应式隔离参数效应）
  safety_shell   : 无 g-g-v vs 有 g-g-v                  （用反应式）
  backend        : 纯反应式 vs base+BC 学生              （需 --bc-ckpt）
  train_paradigm : BC vs DAgger                          （需 --bc-ckpt 与 --dagger-ckpt）

用法：
  python3 -m slash_distill.run_ablations --bc-ckpt ckpt/student_best.pt \
      --dagger-ckpt ckpt_dagger/student_best.pt --episodes 10 --out ablation/results.csv
  # smoke-test（仅反应式行，无 ckpt）：
  python3 -m slash_distill.run_ablations --episodes 1 --max-steps 60 --maps vegas --out /tmp/abl.csv
"""
import argparse
import csv
import os

from .params import GroundTruthLimits, load_distill
from .eval_closed_loop import evaluate

METRIC_KEYS = ["collision_rate", "completion_rate", "mean_lap_count", "mean_steps",
               "mean_speed", "mean_inference_ms", "fallback_rate", "shield_clip_rate"]


def _load_policy(ckpt, device="cpu"):
    if not ckpt or not os.path.isfile(ckpt):
        return None
    from .models.policy import TorchStudentPolicy
    return TorchStudentPolicy(ckpt, device=device)


def main(argv=None):
    ap = argparse.ArgumentParser(description="W8 消融矩阵")
    ap.add_argument("--config", default="distill.yaml")
    ap.add_argument("--bc-ckpt", default=None)
    ap.add_argument("--dagger-ckpt", default=None)
    ap.add_argument("--maps", nargs="*", default=["vegas"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--num-agents", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="ablation/results.csv")
    args = ap.parse_args(argv)

    limits = GroundTruthLimits.from_yaml()
    dcfg = load_distill(args.config)
    bc = _load_policy(args.bc_ckpt, args.device)
    dagger = _load_policy(args.dagger_ckpt, args.device)

    # (ablation, variant, policy, grounding, use_shield) —— policy 用占位字符串，运行时映射
    runs = [
        ("vehicle_params", "default", "reactive", "default", True),
        ("vehicle_params", "sysid",   "reactive", "sysid",   True),
        ("safety_shell",   "no_shield", "reactive", "sysid", False),
        ("safety_shell",   "shield",    "reactive", "sysid", True),
    ]
    if bc is not None:
        runs += [
            ("backend", "reactive",   "reactive", "sysid", True),
            ("backend", "bc_student", "bc",       "sysid", True),
        ]
    if bc is not None and dagger is not None:
        runs += [
            ("train_paradigm", "bc",     "bc",     "sysid", True),
            ("train_paradigm", "dagger", "dagger", "sysid", True),
        ]
    pol_map = {"reactive": None, "bc": bc, "dagger": dagger}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    for ablation, variant, pol_key, grounding, shield in runs:
        policy = pol_map[pol_key]
        for mp in args.maps:
            for sd in args.seeds:
                m = evaluate(policy, limits, dcfg, map_name=mp, grounding=grounding,
                             n_episodes=args.episodes, max_steps=args.max_steps,
                             use_shield=shield, use_fallback=(policy is not None),
                             num_agents=args.num_agents, seed=sd)
                row = {"ablation": ablation, "variant": variant, "map": mp, "seed": sd,
                       "grounding": grounding, "shield": int(shield)}
                row.update({k: round(m[k], 5) for k in METRIC_KEYS})
                rows.append(row)
                print(f"[abl] {ablation}/{variant} map={mp} seed={sd} "
                      f"coll={m['collision_rate']:.2f} speed={m['mean_speed']:.2f} "
                      f"laps={m['mean_lap_count']:.2f} inf={m['mean_inference_ms']:.2f}ms")

    fields = ["ablation", "variant", "map", "seed", "grounding", "shield"] + METRIC_KEYS
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[abl] 写出 {len(rows)} 行 -> {args.out}")


if __name__ == "__main__":
    main()
