"""中心线/赛车线 waypoints 加载（特权专家用，属"训练时可用、部署丢弃"的特权信息）。

约定（与 f1tenth 常见格式兼容）：
  - CSV：每行 x,y[,v][,...]；'#' 注释行跳过；分隔符 ',' 或 ';' 或空白自动识别。
  - 自动在地图同目录找 `<map>_centerline.csv` / `<map>.csv` / `<map>_raceline.csv`。

找不到 waypoints 时返回 None，专家会退化为反应式（仍可跑通流程；用户放图+中心线后自动启用 pure-pursuit）。
"""
import os
import numpy as np


def _sniff_delim(line: str) -> str:
    for d in [",", ";"]:
        if d in line:
            return d
    return None  # 空白分隔


def load_waypoints(path: str):
    """读 CSV → (N,3) [x,y,v]；无 v 列时 v=NaN（专家用 g-g-v 自算）。"""
    if path is None or not os.path.isfile(path):
        return None
    rows = []
    delim = None
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if delim is None:
                delim = _sniff_delim(s)
            parts = s.split(delim) if delim else s.split()
            try:
                vals = [float(p) for p in parts if p != ""]
            except ValueError:
                continue  # 表头
            if len(vals) >= 2:
                x, y = vals[0], vals[1]
                v = vals[2] if len(vals) >= 3 else float("nan")
                rows.append((x, y, v))
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64)


def find_centerline_csv(map_name: str):
    """map_name 可能是 'vegas'（gym 自带）或绝对路径（去/不去 .yaml）。"""
    if map_name is None:
        return None
    base = map_name[:-5] if map_name.endswith(".yaml") else map_name
    cands = [base + "_centerline.csv", base + ".csv", base + "_raceline.csv",
             base + "_wpt.csv"]
    # gym 自带地图目录
    try:
        import f110_gym, glob
        gymdir = os.path.join(os.path.dirname(f110_gym.__file__), "envs", "maps")
        cands += [os.path.join(gymdir, os.path.basename(base) + "_centerline.csv"),
                  os.path.join(gymdir, os.path.basename(base) + ".csv")]
    except Exception:
        pass
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


class WaypointPath:
    """waypoints 的最近点查询 + 局部曲率（纯 numpy，无 scipy 依赖）。"""

    def __init__(self, wpts: np.ndarray):
        self.xy = np.asarray(wpts[:, :2], dtype=np.float64)
        self.v = np.asarray(wpts[:, 2], dtype=np.float64) if wpts.shape[1] >= 3 else \
            np.full(len(wpts), np.nan)
        self.n = len(self.xy)
        # 弧长
        d = np.linalg.norm(np.diff(self.xy, axis=0, append=self.xy[:1]), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(d)[:-1]])

    def nearest_idx(self, x: float, y: float) -> int:
        d2 = (self.xy[:, 0] - x) ** 2 + (self.xy[:, 1] - y) ** 2
        return int(np.argmin(d2))

    def lookahead_point(self, x: float, y: float, Ld: float):
        i0 = self.nearest_idx(x, y)
        acc = 0.0
        i = i0
        for _ in range(self.n):
            j = (i + 1) % self.n
            seg = np.linalg.norm(self.xy[j] - self.xy[i])
            acc += seg
            i = j
            if acc >= Ld:
                break
        return self.xy[i], i

    def curvature_at(self, i: int) -> float:
        a = self.xy[(i - 1) % self.n]
        b = self.xy[i]
        c = self.xy[(i + 1) % self.n]
        ab = b - a
        bc = c - b
        # 用三点外接圆曲率
        area2 = abs(ab[0] * bc[1] - ab[1] * bc[0])
        la = np.linalg.norm(b - a)
        lb = np.linalg.norm(c - b)
        lc = np.linalg.norm(c - a)
        denom = la * lb * lc
        if denom < 1e-9:
            return 0.0
        return 2.0 * area2 / denom
