"""学生观测构造（部署 = 训练同构，单点定义，杜绝漂移）。

观测 = lidar 图（n_layers 通道 × n_beams 束，重采样到定长、按 range_max 归一化）
       + scalar 向量（ego_speed 归一化 + opponent_state[10] 归一化，可选）

- 仿真：n_layers=1，layer = gym 2D scan（1080@270°）重采样到 n_beams。
- 实车：n_layers=3，layers = /scan_3d 的 low/body/high（360°@0.0043）重采样到同一 n_beams。

⚠ 诚实：f110_gym 是 2D，仿真里多层会退化（各层相同）。学生网络层数无关，
   真正的「多层收益」在实车/定性消融里体现。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# opponent_state 布局（与 W4 obstacle_tracker / W4 使用指南一致）
OPP_DIM = 10
# [valid, range, bearing, x, y, vx, vy, speed, range_rate, is_dynamic]


@dataclass
class ObsConfig:
    n_beams: int = 108
    n_layers: int = 1
    fov_rad: float = 4.7
    range_max: float = 20.0
    use_opponent: bool = True
    v_max: float = 2.0
    frame_stack: int = 1

    @classmethod
    def from_distill(cls, dcfg: dict, n_layers: int) -> "ObsConfig":
        o = dcfg.get("obs", {})
        a = dcfg.get("action", {})
        return cls(
            n_beams=int(o.get("n_beams", 108)),
            n_layers=int(n_layers),
            fov_rad=float(o.get("fov_rad", 4.7)),
            range_max=float(o.get("range_max", 20.0)),
            use_opponent=bool(o.get("use_opponent", True)),
            v_max=float(a.get("v_max", 2.0)),
            frame_stack=int(o.get("frame_stack", 1)),
        )

    @property
    def scalar_dim(self) -> int:
        return 1 + (OPP_DIM if self.use_opponent else 0)

    @property
    def lidar_shape(self) -> Tuple[int, int]:
        return (self.n_layers * self.frame_stack, self.n_beams)


def resample_scan(ranges, angle_min: float, angle_increment: float,
                  n_beams: int, fov_rad: float, range_max: float) -> np.ndarray:
    """把任意几何的一帧 scan，按角度线性插值到 [-fov/2, +fov/2] 的 n_beams 定长向量。"""
    ranges = np.asarray(ranges, dtype=np.float64)
    n = ranges.shape[0]
    if n == 0:
        return np.full(n_beams, range_max, dtype=np.float32)
    src_ang = angle_min + np.arange(n) * angle_increment
    r = ranges.copy()
    r[~np.isfinite(r)] = range_max
    r = np.clip(r, 0.0, range_max)
    tgt_ang = np.linspace(-fov_rad / 2.0, fov_rad / 2.0, n_beams)
    # 源角度需单调递增以便插值
    order = np.argsort(src_ang)
    out = np.interp(tgt_ang, src_ang[order], r[order], left=range_max, right=range_max)
    return out.astype(np.float32)


def normalize_opponent(opp: Optional[np.ndarray], range_max: float, v_max: float) -> np.ndarray:
    """opponent_state[10] → 归一化。None / 长度不符 → 全零（valid=0）。"""
    out = np.zeros(OPP_DIM, dtype=np.float32)
    if opp is None:
        return out
    opp = np.asarray(opp, dtype=np.float64).ravel()
    if opp.shape[0] < OPP_DIM:
        return out
    rm = max(range_max, 1e-6)
    vm = max(v_max, 1e-6)
    out[0] = opp[0]                       # valid
    out[1] = opp[1] / rm                  # range
    out[2] = opp[2] / np.pi               # bearing
    out[3] = opp[3] / rm                  # x
    out[4] = opp[4] / rm                  # y
    out[5] = opp[5] / vm                  # vx
    out[6] = opp[6] / vm                  # vy
    out[7] = opp[7] / vm                  # speed
    out[8] = opp[8] / vm                  # range_rate
    out[9] = opp[9]                       # is_dynamic
    return out


def build_obs(layers: List[Tuple[np.ndarray, float, float]],
              ego_speed: float,
              opponent_state: Optional[np.ndarray],
              cfg: ObsConfig) -> Tuple[np.ndarray, np.ndarray]:
    """构造单帧观测。

    layers: list，每元素 = (ranges, angle_min, angle_increment)。长度应 == cfg.n_layers，
            不足则用最后一层补齐（仿真单层 → 复制；实车传 3 层）。
    返回 (lidar: float32 (n_layers, n_beams), scalar: float32 (scalar_dim,))。
         注：frame_stack>1 的历史堆叠由调用方在时间维负责拼接。
    """
    if len(layers) == 0:
        lidar = np.full((cfg.n_layers, cfg.n_beams), 1.0, dtype=np.float32)
    else:
        chans = []
        for i in range(cfg.n_layers):
            src = layers[i] if i < len(layers) else layers[-1]
            ranges, amin, ainc = src
            v = resample_scan(ranges, amin, ainc, cfg.n_beams, cfg.fov_rad, cfg.range_max)
            chans.append(v / max(cfg.range_max, 1e-6))   # 归一化到 ~[0,1]
        lidar = np.stack(chans, axis=0).astype(np.float32)

    scalar = [float(ego_speed) / max(cfg.v_max, 1e-6)]
    if cfg.use_opponent:
        scalar.extend(normalize_opponent(opponent_state, cfg.range_max, cfg.v_max).tolist())
    scalar = np.asarray(scalar, dtype=np.float32)
    return lidar, scalar
