"""slash_distill — W5-W8 特权蒸馏竞速（sysid 接地仿真 + BC/DAgger 学生 + 部署 + 消融）。

导入本包时自动设置 NUMBA_CACHE_DIR：f110_gym 的 dynamic_models 用 @njit(cache=True)，
在某些路径下会报 "cannot cache function ...: no locator available"。指向一个可写目录即可规避。
"""
import os as _os

if not _os.environ.get("NUMBA_CACHE_DIR"):
    _cache = _os.path.join(_os.path.expanduser("~"), ".cache", "slash_distill_numba")
    try:
        _os.makedirs(_cache, exist_ok=True)
        _os.environ["NUMBA_CACHE_DIR"] = _cache
    except OSError:
        _os.environ["NUMBA_CACHE_DIR"] = "/tmp/slash_distill_numba"
        _os.makedirs(_os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

__all__ = ["params"]
