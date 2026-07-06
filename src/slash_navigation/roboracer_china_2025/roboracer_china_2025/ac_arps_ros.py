"""AC-ARPS 安全层的 ROS2 侧胶水：参数声明、实例构建、调试消息。

两份 battle_fast2 节点（包内与包根目录的独立副本）共用本模块，
保证参数名、默认值和调试数组格式完全一致。

部署阶段与参数对应关系（见 修改汇总.md / 论文实验设计）：
- 阶段 0（影子模式）  shield_shadow_mode=True  只记录建议，不改输出
- 阶段 1（只减速）    shield_shadow_mode=False shield_speed_only=True
- 阶段 2（转角残差）  shield_speed_only=False  shield_max_steering_residual 先 0.07 再 0.105
- 阶段 3（在线保形）  shield_adaptive_margin_enabled=True
"""

import math

from std_msgs.msg import Float32MultiArray

try:
    from .ac_arps_core import AdaptiveConformalAckermannShield, debug_to_array
except ImportError:
    from ac_arps_core import AdaptiveConformalAckermannShield, debug_to_array


# 参数名 → 默认值。类型由默认值决定（Foxy declare_parameter 按默认值推断类型）。
SHIELD_PARAM_DEFAULTS = (
    # --- 模式开关 ---
    ('shield_enabled', True),
    ('shield_shadow_mode', True),          # 阶段 0 默认影子模式，最安全
    ('shield_speed_only', True),           # 阶段 1 只允许减速
    ('shield_debug_topic', '/battle_fast2/ac_arps_debug'),
    # --- 车辆几何（务必按实车实测修改）---
    ('shield_wheelbase', 0.33),
    ('shield_vehicle_length', 0.50),
    ('shield_vehicle_width', 0.30),
    ('shield_laser_x_offset', 0.12),
    # --- 预测 ---
    ('shield_prediction_dt', 0.05),
    ('shield_min_horizon', 0.40),
    ('shield_max_horizon', 0.85),
    ('shield_tau_delay', 0.16),
    ('shield_a_brake', 2.5),
    ('shield_a_accel', 2.0),
    # --- 分级介入 ---
    ('shield_max_steering_residual', 0.105),
    ('shield_residual_step', 0.035),
    # --- 安全余量 ---
    ('shield_fixed_margin', 0.08),
    ('shield_adaptive_margin_enabled', False),
    ('shield_target_miscoverage', 0.05),
    ('shield_conformal_window', 200),
    ('shield_conformal_min_samples', 30),
    ('shield_max_conformal_margin', 0.30),
    # --- TTC ---
    ('shield_ttc_emergency', 0.35),
    ('shield_ttc_slow', 1.20),
    ('shield_ttc_percentile', 20.0),
    ('shield_ttc_sector_half_width', 0.35),
    # --- 扫描预处理 ---
    ('shield_max_scan_points', 90),
    ('shield_roi_half_angle', 2.4),
    ('shield_roi_radius', 6.0),
    ('shield_min_valid_beams', 10),
    ('shield_degraded_speed_cap', 0.5),
    # --- 输出限幅 ---
    ('shield_max_speed', 2.7),
    ('shield_max_steer', math.pi / 4.0),
)


def declare_shield_parameters(node):
    """在节点上声明全部 shield_* 参数（可被 launch / yaml 覆盖）。"""
    for name, default in SHIELD_PARAM_DEFAULTS:
        node.declare_parameter(name, default)


def _p(node, name):
    return node.get_parameter(name).value


def build_shield_from_parameters(node):
    """按节点参数构建安全层实例；shield_enabled=False 时返回 None。"""
    if not bool(_p(node, 'shield_enabled')):
        return None
    return AdaptiveConformalAckermannShield(
        wheelbase=float(_p(node, 'shield_wheelbase')),
        vehicle_length=float(_p(node, 'shield_vehicle_length')),
        vehicle_width=float(_p(node, 'shield_vehicle_width')),
        laser_x_offset=float(_p(node, 'shield_laser_x_offset')),
        prediction_dt=float(_p(node, 'shield_prediction_dt')),
        min_horizon=float(_p(node, 'shield_min_horizon')),
        max_horizon=float(_p(node, 'shield_max_horizon')),
        tau_delay=float(_p(node, 'shield_tau_delay')),
        a_brake=float(_p(node, 'shield_a_brake')),
        a_accel=float(_p(node, 'shield_a_accel')),
        max_steering_residual=float(_p(node, 'shield_max_steering_residual')),
        residual_step=float(_p(node, 'shield_residual_step')),
        fixed_margin=float(_p(node, 'shield_fixed_margin')),
        adaptive_margin_enabled=bool(_p(node, 'shield_adaptive_margin_enabled')),
        target_miscoverage=float(_p(node, 'shield_target_miscoverage')),
        conformal_window=int(_p(node, 'shield_conformal_window')),
        conformal_min_samples=int(_p(node, 'shield_conformal_min_samples')),
        max_conformal_margin=float(_p(node, 'shield_max_conformal_margin')),
        ttc_emergency=float(_p(node, 'shield_ttc_emergency')),
        ttc_slow=float(_p(node, 'shield_ttc_slow')),
        ttc_percentile=float(_p(node, 'shield_ttc_percentile')),
        ttc_sector_half_width=float(_p(node, 'shield_ttc_sector_half_width')),
        max_scan_points=int(_p(node, 'shield_max_scan_points')),
        roi_half_angle=float(_p(node, 'shield_roi_half_angle')),
        roi_radius=float(_p(node, 'shield_roi_radius')),
        min_valid_beams=int(_p(node, 'shield_min_valid_beams')),
        degraded_speed_cap=float(_p(node, 'shield_degraded_speed_cap')),
        max_speed=float(_p(node, 'shield_max_speed')),
        max_steer=float(_p(node, 'shield_max_steer')),
        shadow_mode=bool(_p(node, 'shield_shadow_mode')),
        speed_only=bool(_p(node, 'shield_speed_only')),
    )


def make_shield_debug_msg(shield_debug):
    """把 filter() 的 debug dict 转为 Float32MultiArray（字段序见 DEBUG_ARRAY_FIELDS）。"""
    msg = Float32MultiArray()
    msg.data = [float(v) for v in debug_to_array(shield_debug)]
    return msg
