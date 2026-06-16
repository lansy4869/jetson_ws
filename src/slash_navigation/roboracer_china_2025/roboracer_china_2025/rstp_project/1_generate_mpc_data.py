#!/usr/bin/env python3
"""Step 1: Offline MPC data generation for RSTP reproduction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import casadi as ca
import numpy as np


def generate_mpc_trajectory(
    start_state: np.ndarray,
    ref_path: np.ndarray,
    dynamic_obs: Optional[np.ndarray] = None,
    L: int = 128,
    dt: float = 0.1,
) -> Optional[np.ndarray]:
    """
    Reproduce Eq.(1): kinematic MPC tracking + obstacle avoidance.

    Args:
        start_state: [x, y, v, phi]
        ref_path: [x, y, phi], shape (3, Np_ref)
        dynamic_obs: optional obstacle prediction, shape (2, Np_obs)
        L: output trajectory horizon for diffusion model
        dt: discretization time step

    Returns:
        tau: [x, y, qz, qw], shape (4, L), or None if optimization fails
    """
    opti = ca.Opti()
    Np = 40
    lw = 0.33  # wheelbase for F1TENTH

    # State zeta=[x,y,v,phi], control u=[delta,a]
    zeta = opti.variable(4, Np)
    u = opti.variable(2, Np - 1)

    # Cost weights in Eq.(1)
    Q1 = np.diag([10.0, 10.0, 1.0, 0.0])
    R1 = np.diag([0.5, 0.5])
    R2 = np.diag([0.1, 0.1])

    cost = 0
    opti.subject_to(zeta[:, 0] == start_state)

    for k in range(Np - 1):
        x, y, v, phi = zeta[0, k], zeta[1, k], zeta[2, k], zeta[3, k]
        delta, a = u[0, k], u[1, k]

        # 1) Kinematic bicycle model (Euler discretization)
        opti.subject_to(zeta[0, k + 1] == x + v * ca.cos(phi) * dt)
        opti.subject_to(zeta[1, k + 1] == y + v * ca.sin(phi) * dt)
        opti.subject_to(zeta[2, k + 1] == v + a * dt)
        opti.subject_to(zeta[3, k + 1] == phi + (v / lw) * ca.tan(delta) * dt)

        # 2) Reference tracking term
        if k < ref_path.shape[1]:
            ref_k = ca.DM([ref_path[0, k], ref_path[1, k], 1.0, ref_path[2, k]])
            err = zeta[:, k] - ref_k
            cost += ca.mtimes([err.T, Q1, err])

        # 3) Control regularization
        cost += ca.mtimes([u[:, k].T, R2, u[:, k]])
        if k > 0:
            du = u[:, k] - u[:, k - 1]
            cost += ca.mtimes([du.T, R1, du])

        # 4) Dynamic obstacle avoidance term
        if dynamic_obs is not None and k < dynamic_obs.shape[1]:
            gamma = 1.0
            obs_pos = dynamic_obs[:, k]
            dist_sq = (x - obs_pos[0]) ** 2 + (y - obs_pos[1]) ** 2
            cost += gamma * (1.0 / (ca.sqrt(dist_sq) + 1e-3))

    # Hard constraints
    opti.subject_to(opti.bounded(-0.4, u[0, :], 0.4))  # steering
    opti.subject_to(opti.bounded(-2.0, u[1, :], 2.0))  # acceleration
    opti.subject_to(opti.bounded(0.0, zeta[2, :], 5.0))  # speed

    opti.minimize(cost)
    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0, "sb": "yes"})

    try:
        sol = opti.solve()
        zeta_opt = sol.value(zeta)

        # Convert to tau=[x,y,qz,qw], use tail padding to length L
        tau = np.zeros((4, L), dtype=np.float32)
        for i in range(L):
            idx = min(i, Np - 1)
            yaw = zeta_opt[3, idx]
            tau[0, i] = zeta_opt[0, idx]
            tau[1, i] = zeta_opt[1, idx]
            tau[2, i] = np.sin(yaw / 2.0)  # qz
            tau[3, i] = np.cos(yaw / 2.0)  # qw
        return tau
    except Exception:
        return None


def main() -> None:
    print("Generating simulation expert datasets...")

    project_dir = Path(__file__).resolve().parent
    (project_dir / "datasets" / "static").mkdir(parents=True, exist_ok=True)
    (project_dir / "datasets" / "dynamic").mkdir(parents=True, exist_ok=True)

    # Placeholder random data for end-to-end pipeline demo.
    # Replace this with map traversal + MPC rollouts for real reproduction.
    st_data = np.random.randn(1045, 4, 128).astype(np.float32)
    dy_data = np.random.randn(1045, 4, 128).astype(np.float32)
    st_obs = np.random.randn(1045, 2, 128).astype(np.float32)
    dy_obs = np.random.randn(1045, 2, 128).astype(np.float32)

    np.save(project_dir / "datasets" / "static" / "tau.npy", st_data)
    np.save(project_dir / "datasets" / "static" / "obs.npy", st_obs)
    np.save(project_dir / "datasets" / "dynamic" / "tau.npy", dy_data)
    np.save(project_dir / "datasets" / "dynamic" / "obs.npy", dy_obs)
    print("Done: datasets saved under ./datasets/static and ./datasets/dynamic")


if __name__ == "__main__":
    main()
