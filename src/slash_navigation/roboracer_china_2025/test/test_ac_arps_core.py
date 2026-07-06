"""AC-ARPS 安全层核心单元测试（不依赖 ROS）。

场景覆盖：
- 第 0 级：开阔场景基础指令逐位透传；
- 第 1 级：前方障碍触发只减速、转角保持；
- 第 2 级：减速不足以避障时的小幅转角残差；
- 第 3 级：无安全候选时紧急制动；
- 方向性 TTC：接近型障碍触发限速；
- 在线保形余量：预测乐观残差累计后余量上升、静态环境为零；
- 影子模式：只建议不干预；
- 感知降级、调试数组、真实规模扫描的计算耗时冒烟。
"""

import math
from pathlib import Path

import numpy as np

from roboracer_china_2025.ac_arps_core import (
    AdaptiveConformalAckermannShield,
    DEBUG_ARRAY_FIELDS,
    LEVEL_EMERGENCY,
    LEVEL_PASS,
    LEVEL_RESIDUAL,
    LEVEL_SPEED,
    debug_to_array,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# 与默认构造参数一致的扫描规格（±90°、1°/束，模拟 battle_fast2 的前向窗口）
ANGLE_MIN = -math.pi / 2
ANGLE_INC = math.radians(1.0)
N_BEAMS = 181
RANGE_MIN = 0.05
RANGE_MAX = 12.0
LASER_X_OFFSET = 0.12


def make_scan(background=np.inf):
    """构造一帧扫描；背景 inf 表示量程内无回波（自由空间）。"""
    return np.full(N_BEAMS, background, dtype=np.float64)


def scan_angles():
    return ANGLE_MIN + np.arange(N_BEAMS) * ANGLE_INC


def add_point_obstacle(ranges, x_rear_axle, y, width_beams=3):
    """在后轴系 (x, y) 处放一个小障碍（换算回雷达系距离/角度）。"""
    lx = x_rear_axle - LASER_X_OFFSET
    r = math.hypot(lx, y)
    ang = math.atan2(y, lx)
    idx = int(round((ang - ANGLE_MIN) / ANGLE_INC))
    half = width_beams // 2
    for i in range(idx - half, idx + half + 1):
        if 0 <= i < N_BEAMS:
            ranges[i] = min(ranges[i], r)
    return ranges


def add_wall(ranges, distance_from_laser, half_width_deg=40):
    """在雷达正前方 distance 处放一面横墙（每束按几何投影距离）。"""
    angles = scan_angles()
    mask = np.abs(angles) <= math.radians(half_width_deg)
    with np.errstate(divide="ignore"):
        ranges[mask] = np.minimum(ranges[mask], distance_from_laser / np.cos(angles[mask]))
    return ranges


def make_shield(**overrides):
    params = dict(
        wheelbase=0.33,
        vehicle_length=0.50,
        vehicle_width=0.30,
        laser_x_offset=LASER_X_OFFSET,
        prediction_dt=0.05,
        min_horizon=0.40,
        max_horizon=0.85,
        tau_delay=0.16,
        a_brake=2.5,
        fixed_margin=0.08,
        adaptive_margin_enabled=False,
        shadow_mode=False,
        speed_only=True,
        max_speed=2.7,
    )
    params.update(overrides)
    return AdaptiveConformalAckermannShield(**params)


def run_filter(shield, ranges, base_speed, base_steering, stamp=10.0):
    return shield.filter(
        ranges=ranges,
        angle_min=ANGLE_MIN,
        angle_increment=ANGLE_INC,
        range_min=RANGE_MIN,
        range_max=RANGE_MAX,
        base_speed=base_speed,
        base_steering=base_steering,
        stamp_sec=stamp,
    )


# ---------------------------------------------------------------------------
# 第 0 级：透传
# ---------------------------------------------------------------------------

def test_open_scan_passes_command_through_unchanged():
    shield = make_shield()
    shield.prime(1.5, 0.05)
    ranges = make_scan(background=8.0)

    v, d, debug = run_filter(shield, ranges, base_speed=1.5, base_steering=0.05)

    assert debug["intervention_level"] == LEVEL_PASS
    # 一位小数都不修改：逐位相等
    assert v == 1.5
    assert d == 0.05
    assert not debug["applied"]


def test_stop_or_reverse_command_passes_through():
    shield = make_shield()
    ranges = make_scan(background=0.4)  # 极近的墙也不干预停车指令

    v, d, debug = run_filter(shield, ranges, base_speed=0.0, base_steering=0.2)

    assert debug["intervention_level"] == LEVEL_PASS
    assert v == 0.0
    assert d == 0.2


# ---------------------------------------------------------------------------
# 第 1 级：只减速
# ---------------------------------------------------------------------------

def test_close_wall_triggers_speed_only_intervention():
    shield = make_shield()
    shield.prime(2.5, 0.0)
    ranges = add_wall(make_scan(background=9.0), distance_from_laser=0.9)

    v, d, debug = run_filter(shield, ranges, base_speed=2.5, base_steering=0.0)

    assert debug["intervention_level"] in (LEVEL_SPEED, LEVEL_EMERGENCY)
    assert v < 2.5
    assert d == 0.0  # speed_only：转角绝不修改
    assert debug["applied"]


def test_speed_only_never_touches_steering_even_in_emergency():
    shield = make_shield()
    shield.prime(2.5, 0.1)
    ranges = add_wall(make_scan(background=9.0), distance_from_laser=0.35)

    v, d, debug = run_filter(shield, ranges, base_speed=2.5, base_steering=0.1)

    assert debug["intervention_level"] == LEVEL_EMERGENCY
    assert v == 0.0
    assert d == 0.1


# ---------------------------------------------------------------------------
# 第 2 级：小幅转角残差
# ---------------------------------------------------------------------------

def _residual_scenario_shield():
    # 弱制动 + 零延迟：让"只减速"在时域内依旧撞上，残差转向才能避开
    return make_shield(
        speed_only=False,
        tau_delay=0.0,
        a_brake=0.3,
        min_horizon=0.85,
        max_horizon=0.85,
    )


def test_marginal_obstacle_resolved_by_small_steering_residual():
    shield = _residual_scenario_shield()
    shield.prime(2.0, 0.0)
    # 障碍在正前方略偏左：直行擦碰，右打小残差即可绕开
    ranges = add_point_obstacle(make_scan(background=9.0), x_rear_axle=1.2, y=0.13)

    v, d, debug = run_filter(shield, ranges, base_speed=2.0, base_steering=0.0)

    assert debug["intervention_level"] == LEVEL_RESIDUAL
    assert abs(d - 0.0) <= shield.max_steering_residual + 1e-9
    assert d < 0.0  # 向右残差（远离左侧障碍）
    assert debug["predicted_clearance"] > 0.0


def test_residual_magnitude_never_exceeds_configured_bound():
    shield = _residual_scenario_shield()
    shield.prime(2.0, 0.0)
    for y in (0.10, 0.13, 0.16, -0.10, -0.13, -0.16):
        ranges = add_point_obstacle(make_scan(background=9.0), x_rear_axle=1.2, y=y)
        _, d, _ = run_filter(shield, ranges, base_speed=2.0, base_steering=0.0)
        assert abs(d) <= shield.max_steering_residual + 1e-9


# ---------------------------------------------------------------------------
# 方向性 TTC
# ---------------------------------------------------------------------------

def test_approaching_wall_triggers_ttc_speed_cap():
    shield = make_shield()
    shield.prime(1.5, 0.0)

    frame1 = add_wall(make_scan(background=9.0), distance_from_laser=3.0)
    v1, d1, debug1 = run_filter(shield, frame1, 1.5, 0.0, stamp=10.0)
    assert debug1["intervention_level"] == LEVEL_PASS  # 首帧无 TTC，几何净空充足

    # 0.1 s 内推进 0.3 m → closing ≈ 3 m/s → TTC ≈ 0.75 s < ttc_slow
    frame2 = add_wall(make_scan(background=9.0), distance_from_laser=2.7)
    v2, d2, debug2 = run_filter(shield, frame2, 1.5, 0.0, stamp=10.1)

    assert debug2["ttc"] < shield.ttc_slow
    assert debug2["intervention_level"] == LEVEL_SPEED
    assert "ttc" in debug2["intervention_reason"]
    assert v2 < 1.5
    assert d2 == 0.0


def test_static_wall_far_away_gives_no_ttc_intervention():
    shield = make_shield()
    shield.prime(1.5, 0.0)
    frame = add_wall(make_scan(background=9.0), distance_from_laser=3.0)
    run_filter(shield, frame, 1.5, 0.0, stamp=10.0)
    v, d, debug = run_filter(shield, frame.copy(), 1.5, 0.0, stamp=10.1)

    assert math.isinf(debug["ttc"])
    assert debug["intervention_level"] == LEVEL_PASS
    assert (v, d) == (1.5, 0.0)


# ---------------------------------------------------------------------------
# 在线保形余量
# ---------------------------------------------------------------------------

def test_conformal_margin_grows_under_optimistic_prediction():
    shield = make_shield(adaptive_margin_enabled=True, conformal_min_samples=10)
    # 静止车 + 每帧墙面额外逼近 0.05 m（预测持续乐观 0.05 m）
    dist = 3.0
    for k in range(60):
        frame = add_wall(make_scan(background=9.0), distance_from_laser=dist)
        run_filter(shield, frame, base_speed=0.0, base_steering=0.0, stamp=10.0 + 0.05 * k)
        dist -= 0.05

    assert shield.conformal_margin > 0.03
    assert shield.conformal_margin <= shield.max_conformal_margin
    assert shield.effective_radius > shield.circle_radius_geom + shield.fixed_margin


def test_conformal_margin_stays_zero_in_static_environment():
    shield = make_shield(adaptive_margin_enabled=True, conformal_min_samples=10)
    frame = add_wall(make_scan(background=9.0), distance_from_laser=3.0)
    for k in range(60):
        run_filter(shield, frame.copy(), 0.0, 0.0, stamp=10.0 + 0.05 * k)

    assert shield.conformal_margin == 0.0


def test_conformal_margin_disabled_by_default():
    shield = make_shield(adaptive_margin_enabled=False)
    assert shield.conformal_margin == 0.0
    assert shield.effective_radius == shield.circle_radius_geom + shield.fixed_margin


# ---------------------------------------------------------------------------
# 影子模式
# ---------------------------------------------------------------------------

def test_shadow_mode_records_suggestion_but_never_modifies_output():
    shield = make_shield(shadow_mode=True)
    shield.prime(2.5, 0.0)
    ranges = add_wall(make_scan(background=9.0), distance_from_laser=0.9)

    v, d, debug = run_filter(shield, ranges, base_speed=2.5, base_steering=0.0)

    assert (v, d) == (2.5, 0.0)                # 输出恒等于基础指令
    assert debug["intervention_level"] > LEVEL_PASS
    assert debug["suggested_speed"] < 2.5      # 建议值单独记录
    assert not debug["applied"]
    assert debug["shadow_mode"]


# ---------------------------------------------------------------------------
# 感知降级与原始扫描语义
# ---------------------------------------------------------------------------

def test_all_nan_scan_degrades_to_speed_cap():
    shield = make_shield()
    shield.prime(1.5, 0.0)
    ranges = np.full(N_BEAMS, np.nan)

    v, d, debug = run_filter(shield, ranges, base_speed=1.5, base_steering=0.0)

    assert debug["sensor_degraded"]
    assert v <= shield.degraded_speed_cap + 1e-9
    assert d == 0.0


def test_inf_beams_are_free_space_not_obstacles():
    shield = make_shield()
    shield.prime(1.5, 0.0)
    ranges = make_scan(background=np.inf)  # 全部无回波 → 自由空间

    v, d, debug = run_filter(shield, ranges, base_speed=1.5, base_steering=0.0)

    assert not debug["sensor_degraded"]
    assert debug["intervention_level"] == LEVEL_PASS
    assert (v, d) == (1.5, 0.0)


# ---------------------------------------------------------------------------
# 调试数组与计算耗时
# ---------------------------------------------------------------------------

def test_debug_array_is_finite_and_matches_field_order():
    shield = make_shield()
    shield.prime(1.5, 0.0)
    ranges = make_scan(background=np.inf)  # TTC=inf 的场景也必须压成有限值
    _, _, debug = run_filter(shield, ranges, 1.5, 0.0)

    arr = debug_to_array(debug)
    assert len(arr) == len(DEBUG_ARRAY_FIELDS)
    assert all(math.isfinite(x) for x in arr)
    assert arr[0] == float(debug["intervention_level"])


def test_realistic_scan_size_compute_time_smoke():
    # 真实 /scan 规格：360°、约 1460 束（MID-360 → pointcloud_to_laserscan）
    n = 1460
    angle_min = -math.pi
    angle_inc = 2.0 * math.pi / n
    ranges = np.full(n, 6.0)
    ranges[: n // 4] = 1.5   # 造出障碍让分级搜索全部展开
    shield = make_shield(speed_only=False)
    shield.prime(2.5, 0.0)

    for k in range(5):
        _, _, debug = shield.filter(
            ranges=ranges,
            angle_min=angle_min,
            angle_increment=angle_inc,
            range_min=0.05,
            range_max=12.0,
            base_speed=2.5,
            base_steering=0.0,
            stamp_sec=20.0 + 0.05 * k,
        )
    # 宽松冒烟上界（Jetson 实测另行统计 P50/P95）
    assert debug["compute_time_ms"] < 200.0


# ---------------------------------------------------------------------------
# 节点源码集成点检查（与 test_battle_fast2_defaults.py 同风格）
# ---------------------------------------------------------------------------

def test_both_nodes_integrate_shield_at_three_locations():
    for rel in ("battle_fast2_node.py", "roboracer_china_2025/battle_fast2_node.py"):
        source = (PACKAGE_ROOT / rel).read_text(encoding="utf-8")
        # 位置一：nan_to_num 调用之前保存原始扫描
        assert "raw_ranges = np.asarray(data.ranges, dtype=np.float32).copy()" in source
        assert source.index("raw_ranges = np.asarray") < source.index("np.nan_to_num(")
        # 位置二：初始化安全层
        assert "build_shield_from_parameters" in source
        # 位置三：发布前过滤
        assert "apply_predictive_shield" in source


def test_shield_defaults_to_shadow_mode_in_ros_glue():
    source = (
        PACKAGE_ROOT / "roboracer_china_2025/ac_arps_ros.py"
    ).read_text(encoding="utf-8")
    assert "('shield_shadow_mode', True)" in source
    assert "('shield_speed_only', True)" in source
    assert "('shield_adaptive_margin_enabled', False)" in source
