"""无状态帧堆叠 1D-CNN 学生网络（W6，RaceMOP LidarNet 思路，比 GRU 更 TRT 友好）。

输入：
  lidar  (B, C, n_beams)  —— C = n_layers * frame_stack（仿真 C=1；实车 C=3）
  scalar (B, S)           —— [ego_speed_norm] (+ opponent_state[10]_norm)
输出：
  action (B, 2) in [-1,1] —— (steer_norm, speed_norm)，使用时 *steer_max / 反归一化到 [0,v_max]

无状态、无控制流分支 → 直接 torch.onnx.export，TensorRT 友好；前向约 1–2 ms。
"""
from typing import List
import torch
import torch.nn as nn


class StudentCNN(nn.Module):
    def __init__(self,
                 n_beams: int = 108,
                 in_channels: int = 1,
                 scalar_dim: int = 11,
                 conv_channels: List[int] = (32, 64, 64),
                 kernel_sizes: List[int] = (7, 5, 3),
                 strides: List[int] = (3, 2, 2),
                 mlp_hidden: List[int] = (256, 128),
                 dropout: float = 0.0):
        super().__init__()
        self.n_beams = n_beams
        self.in_channels = in_channels
        self.scalar_dim = scalar_dim

        convs = []
        c_in = in_channels
        for c_out, k, st in zip(conv_channels, kernel_sizes, strides):
            convs += [nn.Conv1d(c_in, c_out, kernel_size=k, stride=st, padding=k // 2),
                      nn.ReLU(inplace=True)]
            c_in = c_out
        self.conv = nn.Sequential(*convs)

        # 推断卷积输出展平维度
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, n_beams)
            conv_out = self.conv(dummy).flatten(1).shape[1]
        self.conv_out_dim = conv_out

        mlp = []
        h_in = conv_out + scalar_dim
        for h in mlp_hidden:
            mlp += [nn.Linear(h_in, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                mlp += [nn.Dropout(dropout)]
            h_in = h
        mlp += [nn.Linear(h_in, 2), nn.Tanh()]
        self.head = nn.Sequential(*mlp)

    def forward(self, lidar: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        z = self.conv(lidar).flatten(1)
        z = torch.cat([z, scalar], dim=1)
        return self.head(z)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_student(distill_cfg: dict, in_channels: int, scalar_dim: int) -> StudentCNN:
    o = distill_cfg.get("obs", {})
    m = distill_cfg.get("model", {})
    return StudentCNN(
        n_beams=int(o.get("n_beams", 108)),
        in_channels=in_channels,
        scalar_dim=scalar_dim,
        conv_channels=tuple(m.get("conv_channels", (32, 64, 64))),
        kernel_sizes=tuple(m.get("kernel_sizes", (7, 5, 3))),
        strides=tuple(m.get("strides", (3, 2, 2))),
        mlp_hidden=tuple(m.get("mlp_hidden", (256, 128))),
        dropout=float(m.get("dropout", 0.0)),
    )


# --------- 动作归一化/反归一化（训练标签与部署反解共用） ---------
def normalize_action(steer: float, speed: float, steer_max: float, v_max: float):
    return (steer / max(steer_max, 1e-6), 2.0 * speed / max(v_max, 1e-6) - 1.0)


def denormalize_action(steer_n: float, speed_n: float, steer_max: float, v_max: float):
    steer = steer_n * steer_max
    speed = (speed_n + 1.0) * 0.5 * v_max
    return steer, speed
