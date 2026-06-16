"""W8：把 run_ablations 的 CSV 汇总成论文用的表（markdown）+ 柱状图（可选 matplotlib）。

用法：
  python3 -m slash_distill.make_figures --csv ablation/results.csv --out-dir ablation
产出：
  ablation/summary.md  —— 各 (ablation,variant) 的指标均值表（始终生成）
  ablation/*.png       —— 每个 ablation 一张对照柱状图（装了 matplotlib 才生成）
"""
import argparse
import csv
import os
from collections import defaultdict

PLOT_METRICS = ["collision_rate", "mean_speed", "mean_lap_count",
                "shield_clip_rate", "fallback_rate", "mean_inference_ms"]


def _read(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _aggregate(rows):
    """(ablation,variant) -> {metric: mean}。"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["ablation"], r["variant"])
        for m in PLOT_METRICS:
            try:
                acc[key][m].append(float(r[m]))
            except (KeyError, ValueError):
                pass
    out = {}
    for key, md in acc.items():
        out[key] = {m: (sum(v) / len(v) if v else float("nan")) for m, v in md.items()}
    return out


def _write_md(agg, out_dir):
    path = os.path.join(out_dir, "summary.md")
    by_abl = defaultdict(list)
    for (ab, var), md in agg.items():
        by_abl[ab].append((var, md))
    lines = ["# W8 消融汇总（仿真）\n",
             "> 由 run_ablations.py 的 CSV 自动汇总；感知维度(3D)消融见实车/定性结果。\n"]
    for ab, items in by_abl.items():
        lines.append(f"\n## {ab}\n")
        lines.append("| variant | " + " | ".join(PLOT_METRICS) + " |")
        lines.append("|" + "---|" * (len(PLOT_METRICS) + 1))
        for var, md in items:
            cells = " | ".join(f"{md.get(m, float('nan')):.4f}" for m in PLOT_METRICS)
            lines.append(f"| {var} | {cells} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[fig] 表 -> {path}")
    return path


def _plots(agg, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[fig] 未装 matplotlib，跳过柱状图（pip install matplotlib 可出图）")
        return
    by_abl = defaultdict(list)
    for (ab, var), md in agg.items():
        by_abl[ab].append((var, md))
    for ab, items in by_abl.items():
        variants = [v for v, _ in items]
        fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(3 * len(PLOT_METRICS), 3))
        if len(PLOT_METRICS) == 1:
            axes = [axes]
        for ax, m in zip(axes, PLOT_METRICS):
            vals = [md.get(m, 0.0) for _, md in items]
            ax.bar(variants, vals)
            ax.set_title(m, fontsize=8)
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        fig.suptitle(f"ablation: {ab}")
        fig.tight_layout()
        p = os.path.join(out_dir, f"ablation_{ab}.png")
        fig.savefig(p, dpi=120); plt.close(fig)
        print(f"[fig] 图 -> {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="W8 图表汇总")
    ap.add_argument("--csv", default="ablation/results.csv")
    ap.add_argument("--out-dir", default="ablation")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    rows = _read(args.csv)
    if not rows:
        print(f"[fig] CSV 为空：{args.csv}")
        return
    agg = _aggregate(rows)
    _write_md(agg, args.out_dir)
    _plots(agg, args.out_dir)


if __name__ == "__main__":
    main()
