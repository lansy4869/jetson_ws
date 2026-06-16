#!/usr/bin/env python3
"""Step 2: Conditional diffusion model training for RSTP reproduction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler


class RSTP_Diffuser(nn.Module):
    """1D denoiser with obstacle transformer encoder + CFG masking."""

    def __init__(self, state_dim: int = 4, obs_dim: int = 2, embed_dim: int = 64, seq_len: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=obs_dim,
            nhead=2,
            batch_first=True,
        )
        self.obs_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.obs_proj = nn.Linear(obs_dim * seq_len, embed_dim)

        self.in_conv = nn.Conv1d(state_dim, embed_dim, kernel_size=3, padding=1)
        self.mid = nn.Sequential(
            nn.Mish(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.Mish(),
        )
        self.out_conv = nn.Conv1d(embed_dim, state_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: noisy traj [B, 4, L]
            t: diffusion timestep [B, 1]
            obs: obstacle sequence [B, 2, L]
            mask: CFG mask [B, 1], 1=conditional, 0=unconditional
        """
        t_emb = self.time_mlp(t.float())
        obs_feat = self.obs_transformer(obs.permute(0, 2, 1))  # [B, L, 2]
        obs_emb = self.obs_proj(obs_feat.reshape(x.shape[0], -1)) * mask
        cond_emb = (t_emb + obs_emb).unsqueeze(-1)  # [B, embed_dim, 1]

        h = self.in_conv(x) + cond_emb
        h = self.mid(h)
        return self.out_conv(h)


def train_model(dataset_path: Path, obs_path: Path, save_path: Path, epochs: int = 100, bs: int = 32) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RSTP_Diffuser().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")

    taus = torch.tensor(np.load(dataset_path), dtype=torch.float32, device=device)
    obs = torch.tensor(np.load(obs_path), dtype=torch.float32, device=device)

    model.train()
    for epoch in range(epochs):
        indices = torch.randperm(taus.shape[0], device=device)
        for i in range(0, taus.shape[0], bs):
            batch_idx = indices[i : i + bs]
            batch_traj = taus[batch_idx]
            batch_obs = obs[batch_idx]

            noise = torch.randn_like(batch_traj)
            timesteps = torch.randint(0, 100, (batch_traj.shape[0], 1), device=device)
            noisy_traj = scheduler.add_noise(batch_traj, noise, timesteps.squeeze(-1))

            # CFG: 10% unconditional training
            mask = (torch.rand(batch_traj.shape[0], 1, device=device) > 0.1).float()

            noise_pred = model(noisy_traj, timesteps, batch_obs, mask)
            loss = nn.MSELoss()(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            print(f"[{save_path.stem}] Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to: {save_path}")


def main() -> None:
    root = Path(__file__).resolve().parent
    train_model(
        root / "datasets" / "static" / "tau.npy",
        root / "datasets" / "static" / "obs.npy",
        root / "weights" / "model_st.pth",
    )
    train_model(
        root / "datasets" / "dynamic" / "tau.npy",
        root / "datasets" / "dynamic" / "obs.npy",
        root / "weights" / "model_dy.pth",
    )


if __name__ == "__main__":
    main()
