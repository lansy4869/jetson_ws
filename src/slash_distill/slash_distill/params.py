"""配置加载 + sysid 接地参数容器（被所有离线脚本与部署节点复用）。

设计点：
- 配置文件 `config/sysid_params.yaml` / `config/distill.yaml` 在源码树与 colcon install/share 下都能找到。
- `GroundTruthLimits` 把「本车实测可行域」打包成一个轻量 dataclass，sim 与部署同一套数字。
"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------- 配置路径解析
def _candidate_config_dirs() -> List[str]:
    here = os.path.dirname(os.path.abspath(__file__))           # .../slash_distill/slash_distill
    pkg_root = os.path.dirname(here)                            # .../slash_distill
    cands = [
        os.path.join(pkg_root, "config"),                      # 源码树
        os.path.join(here, "config"),
    ]
    # colcon install: share/slash_distill/config
    try:
        from ament_index_python.packages import get_package_share_directory
        cands.insert(0, os.path.join(get_package_share_directory("slash_distill"), "config"))
    except Exception:
        pass
    # 环境变量覆盖
    env = os.environ.get("SLASH_DISTILL_CONFIG_DIR")
    if env:
        cands.insert(0, env)
    return cands


def find_config(name: str) -> Optional[str]:
    for d in _candidate_config_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def load_yaml(name: str) -> Dict[str, Any]:
    path = find_config(name)
    if path is None:
        raise FileNotFoundError(
            f"找不到配置 {name}；查找路径：{_candidate_config_dirs()}（设 SLASH_DISTILL_CONFIG_DIR 可覆盖）")
    if yaml is None:
        raise ImportError("缺少 pyyaml，请 `pip install pyyaml`")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------- 接地极限容器
@dataclass
class GroundTruthLimits:
    """本车实测可行域（g-g-v）+ 纵向一阶模型。sim 与部署共用。"""
    wheelbase: float = 0.25
    steering_limit_rad: float = 0.34
    ax_accel_max: float = 6.35
    ax_brake_max: float = 6.66
    ay_max: float = 9.81
    ay_source: str = "ASSUMED"
    ay_safety_factor: float = 0.6
    v_max: float = 2.0
    v_min: float = 0.0
    long_gain_K: float = 0.9139
    long_tau_s: float = 0.255
    long_delay_s: float = 0.0
    mu_ground: float = 1.0
    gym_defaults: Dict[str, float] = field(default_factory=lambda: {
        "lf": 0.15875, "lr": 0.17145, "mu": 1.0489})

    @classmethod
    def from_yaml(cls, name: str = "sysid_params.yaml") -> "GroundTruthLimits":
        d = load_yaml(name)
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)


def load_distill(name: str = "distill.yaml") -> Dict[str, Any]:
    return load_yaml(name)
