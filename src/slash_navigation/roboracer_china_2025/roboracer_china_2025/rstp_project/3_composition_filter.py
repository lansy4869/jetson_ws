#!/usr/bin/env python3
"""Step 3: Diffusion composition inference + safety filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from diffusers import DDIMScheduler
import importlib.util


def _load_model_class():
    """Load RSTP_Diffuser from `2_train_diffusion.py` (numeric filename safe)."""
    root = Path(__file__).resolve().parent
    module_path = root / "2_train_diffusion.py"
    spec = importlib.util.spec_from_file_location("train_diffusion_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RSTP_Diffuser


def ddim_compose_inference(
    model_st: torch.nn.Module,
    model_dy: torch.nn.Module,
    obs_st: torch.Tensor,
    obs_dy: torch.Tensor,
    L: int = 128,
    n_filter: int = 8,
    nu_st: float = 2.5,
    nu_dy: float = 5.0,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Eq.(3): test-time diffusion composition with DDIM 8-step acceleration."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_st.eval()
    model_dy.eval()
    scheduler = DDIMScheduler(num_train_timesteps=100)
    scheduler.set_timesteps(8)

    x = torch.randn(n_filter, 4, L, device=device)
    mask_cond = torch.ones(n_filter, 1, device=device)
    mask_uncond = torch.zeros(n_filter, 1, device=device)

    with torch.no_grad():
        for t in scheduler.timesteps:
            t_value = int(t.item()) if torch.is_tensor(t) else int(t)
            t_batch = torch.full((n_filter, 1), t_value, device=device)

            eps_uncond = model_st(x, t_batch, obs_st, mask_uncond)
            eps_st = model_st(x, t_batch, obs_st, mask_cond)
            eps_dy = model_dy(x, t_batch, obs_dy, mask_cond)

            eps_compose = eps_uncond + nu_st * (eps_st - eps_uncond) + nu_dy * (eps_dy - eps_uncond)
            x = scheduler.step(eps_compose, t, x).prev_sample

    return x.detach().cpu().numpy()


def _quat_to_yaw(qz: np.ndarray, qw: np.ndarray) -> np.ndarray:
    return np.arctan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz**2))


def safety_filter(
    batch_trajs: np.ndarray,
    obstacles: Sequence[Tuple[float, float]],
    dt: float = 0.1,
    lw: float = 0.33,
    w_max: float = 2.0,
    safety_threshold: float = 0.08,
) -> Optional[np.ndarray]:
    """Algorithm 1 + Eq.(4,5): pick safe kinematically feasible trajectory."""
    n = batch_trajs.shape[0]
    v_inf = 1e6

    l_costs = []
    a_costs = []
    d_costs = []
    safe_costs = []
    phi_penalties = []

    obs_arr = np.array(obstacles, dtype=np.float32) if len(obstacles) > 0 else np.empty((0, 2), dtype=np.float32)

    for i in range(n):
        tau = batch_trajs[i]  # [4, L]
        q = tau[:2, :]  # [x,y]
        qz, qw = tau[2, :], tau[3, :]
        yaw = _quat_to_yaw(qz, qw)

        dq = q[:, 1:] - q[:, :-1]
        d_i = np.linalg.norm(dq, axis=0)
        v_i = d_i / dt
        a_i = (v_i[1:] - v_i[:-1]) / dt if len(v_i) > 1 else np.array([0.0], dtype=np.float32)
        r_i = (yaw[1:] - yaw[:-1]) / dt

        l_costs.append(float(np.sum(d_i)))
        a_costs.append(float(np.linalg.norm(a_i)))

        if len(r_i) > 1 and len(v_i) > 1:
            safe_v = np.where(v_i[:-1] == 0.0, 1e-5, v_i[:-1])
            delta = np.arctan(lw * r_i[:-1] / safe_v)
            d_costs.append(float(np.linalg.norm(delta)))
        else:
            d_costs.append(0.0)

        if obs_arr.shape[0] > 0:
            traj_xy = q.T  # [L,2]
            dists = np.linalg.norm(traj_xy[None, :, :] - obs_arr[:, None, :], axis=2)
            min_dist = float(np.min(dists))
        else:
            min_dist = 1.0

        if min_dist <= safety_threshold:
            safe_costs.append(v_inf)
        else:
            safe_costs.append(float(1.0 / (np.mean(d_i) + 1.0)))

        if np.any(np.abs(np.diff(yaw)) > w_max * dt):
            phi_penalties.append(v_inf)
        else:
            phi_penalties.append(0.0)

    def normalize(arr: Sequence[float]) -> np.ndarray:
        arr_np = np.array(arr, dtype=np.float64)
        ptp = np.ptp(arr_np)
        return (arr_np - np.min(arr_np)) / (ptp if ptp > 0 else 1e-5)

    j_total = (
        normalize(l_costs)
        + normalize(a_costs)
        + normalize(d_costs)
        + np.array(safe_costs, dtype=np.float64)
        + np.array(phi_penalties, dtype=np.float64)
    )

    best_idx = int(np.argmin(j_total))
    if j_total[best_idx] >= v_inf:
        return None
    return batch_trajs[best_idx]


def main() -> None:
    root = Path(__file__).resolve().parent
    RSTP_Diffuser = _load_model_class()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_st = RSTP_Diffuser().to(device)
    model_dy = RSTP_Diffuser().to(device)

    model_st.load_state_dict(torch.load(root / "weights" / "model_st.pth", map_location=device))
    model_dy.load_state_dict(torch.load(root / "weights" / "model_dy.pth", map_location=device))

    obs_st_mock = torch.zeros(8, 2, 128, device=device)
    obs_dy_mock = torch.zeros(8, 2, 128, device=device)
    obstacles_mock = [(2.0, 2.0)]

    print("Running composed diffusion inference (DDIM 8 steps)...")
    batch_trajs = ddim_compose_inference(model_st, model_dy, obs_st_mock, obs_dy_mock, device=device)

    print("Running safety filter...")
    optimal_traj = safety_filter(batch_trajs, obstacles_mock)

    if optimal_traj is not None:
        print(f"Success: best feasible trajectory found, shape={optimal_traj.shape}")
    else:
        print("All candidates invalid, replanning required.")


if __name__ == "__main__":
    main()
