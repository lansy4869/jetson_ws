"""W6：BC（行为克隆）训练——确定性监督，无奖励、无 RL gap。

目标：min_theta E_{(s,a*)~D_exp} || pi_theta(s) - a* ||^2

用法：
  python3 -m slash_distill.train_bc --data data/demos.npz --epochs 100 --out ckpt
  # smoke-test：
  python3 -m slash_distill.train_bc --data /tmp/demo_smoke.npz --epochs 2 --out /tmp/ckpt_smoke

输出 ckpt/student_best.pt：state_dict + model_cfg + meta(n_layers,n_beams,scalar_dim,steer_max,v_max)
"""
import argparse
import os
import numpy as np

from .params import load_distill
from .models.student_cnn import build_student, normalize_action


def _normalize_actions(actions: np.ndarray, steer_max: float, v_max: float) -> np.ndarray:
    out = np.empty_like(actions)
    out[:, 0] = actions[:, 0] / max(steer_max, 1e-6)
    out[:, 1] = 2.0 * actions[:, 1] / max(v_max, 1e-6) - 1.0
    return np.clip(out, -1.0, 1.0)


def main(argv=None):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    ap = argparse.ArgumentParser(description="W6 BC 训练")
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="distill.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    dcfg = load_distill(args.config)
    tc = dcfg.get("train", {})
    epochs = args.epochs if args.epochs is not None else int(tc.get("epochs", 100))
    bs = args.batch_size if args.batch_size is not None else int(tc.get("batch_size", 256))
    lr = args.lr if args.lr is not None else float(tc.get("lr", 1e-3))
    out_dir = args.out or tc.get("ckpt_dir", "ckpt")
    seed = args.seed if args.seed is not None else int(tc.get("seed", 0))
    want_dev = args.device or tc.get("device", "cuda")
    device = "cuda" if (want_dev == "cuda" and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)

    d = np.load(args.data)
    lidar = d["lidar"].astype(np.float32)
    scalar = d["scalar"].astype(np.float32)
    action = d["action"].astype(np.float32)
    if lidar.shape[0] == 0:
        raise RuntimeError(f"数据集为空：{args.data}")
    steer_max = float(d["steer_max"]); v_max = float(d["v_max"])
    n_layers = int(d["n_layers"]); n_beams = int(d["n_beams"]); scalar_dim = int(d["scalar_dim"])
    y = _normalize_actions(action, steer_max, v_max)

    model = build_student(dcfg, in_channels=n_layers, scalar_dim=scalar_dim).to(device)
    model_cfg = dict(n_beams=n_beams, in_channels=n_layers, scalar_dim=scalar_dim,
                     conv_channels=tuple(dcfg.get("model", {}).get("conv_channels", (32, 64, 64))),
                     kernel_sizes=tuple(dcfg.get("model", {}).get("kernel_sizes", (7, 5, 3))),
                     strides=tuple(dcfg.get("model", {}).get("strides", (3, 2, 2))),
                     mlp_hidden=tuple(dcfg.get("model", {}).get("mlp_hidden", (256, 128))),
                     dropout=float(dcfg.get("model", {}).get("dropout", 0.0)))
    print(f"[train] device={device} params={model.num_params()} frames={lidar.shape[0]} "
          f"lidar={lidar.shape} scalar={scalar.shape}")

    # train/val split
    n = lidar.shape[0]
    idx = np.random.permutation(n)
    val_frac = float(tc.get("val_frac", 0.1))
    n_val = max(1, int(n * val_frac)) if n > 1 else 0
    vi, ti = idx[:n_val], idx[n_val:]

    def loader(ids, shuffle):
        ds = TensorDataset(torch.from_numpy(lidar[ids]), torch.from_numpy(scalar[ids]),
                           torch.from_numpy(y[ids]))
        return DataLoader(ds, batch_size=min(bs, max(1, len(ids))), shuffle=shuffle)

    train_dl = loader(ti, True)
    val_dl = loader(vi, False) if n_val > 0 else None

    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=float(tc.get("weight_decay", 0.0)))
    loss_fn = torch.nn.MSELoss()

    os.makedirs(out_dir, exist_ok=True)
    best = float("inf")
    best_path = os.path.join(out_dir, "student_best.pt")
    for ep in range(1, epochs + 1):
        model.train(); tr = 0.0; nb = 0
        for lb, sb, yb in train_dl:
            lb, sb, yb = lb.to(device), sb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(lb, sb), yb)
            loss.backward(); opt.step()
            tr += float(loss); nb += 1
        tr /= max(nb, 1)
        # val
        vl = tr
        if val_dl is not None:
            model.eval(); vsum = 0.0; vnb = 0
            with torch.no_grad():
                for lb, sb, yb in val_dl:
                    lb, sb, yb = lb.to(device), sb.to(device), yb.to(device)
                    vsum += float(loss_fn(model(lb, sb), yb)); vnb += 1
            vl = vsum / max(vnb, 1)
        if vl < best:
            best = vl
            torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg,
                        "meta": {"n_layers": n_layers, "n_beams": n_beams,
                                 "scalar_dim": scalar_dim, "steer_max": steer_max, "v_max": v_max}},
                       best_path)
        if ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs:
            print(f"[train] epoch {ep}/{epochs} train_mse={tr:.5f} val_mse={vl:.5f} best={best:.5f}")

    print(f"[train] 完成，best val_mse={best:.5f} -> {best_path}")


if __name__ == "__main__":
    main()
