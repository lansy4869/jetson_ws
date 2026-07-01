import math

from roboracer_china_2025.drive_arbiter_core import (
    ArbiterConfig,
    DriveCommand,
    SafetyDiagnostics,
    ShieldConfig,
    SourceHealth,
    arbitrate,
)


def config():
    return ArbiterConfig(
        max_steer=0.4,
        max_speed=2.0,
        min_speed=0.0,
        command_timeout_s=0.2,
        odom_timeout_s=0.3,
        prefer_reactive_on_mpc_timeout=True,
    )


def healthy(stamp_s=1.0):
    return SourceHealth(stamp_s=stamp_s, valid=True)


def shield_config():
    return ShieldConfig(
        enabled=True,
        diag_timeout_s=0.2,
        yellow_margin_m=0.55,
        orange_margin_m=0.35,
        red_margin_m=0.18,
        black_clearance_m=0.30,
        orange_blend=0.5,
    )


def test_selects_mpc_when_all_inputs_are_healthy():
    decision = arbitrate(
        now_s=1.1,
        mpc_command=DriveCommand(0.1, 1.5),
        mpc_health=healthy(1.05),
        reactive_command=DriveCommand(-0.2, 0.8),
        reactive_health=healthy(1.05),
        odom_health=healthy(1.05),
        config=config(),
    )

    assert decision.source == "mpc"
    assert decision.command == DriveCommand(0.1, 1.5)


def test_stops_when_odom_is_stale():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.1, 1.0),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.1, 0.5),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.1),
        config=config(),
    )

    assert decision.source == "stop"
    assert decision.command == DriveCommand(0.0, 0.0)
    assert "odom stale" in decision.reason


def test_reactive_fallback_is_used_when_mpc_command_is_stale():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.1, 1.0),
        mpc_health=healthy(0.0),
        reactive_command=DriveCommand(-0.2, 0.7),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
    )

    assert decision.source == "reactive"
    assert decision.command == DriveCommand(-0.2, 0.7)


def test_non_finite_mpc_command_falls_back_to_reactive():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(math.nan, 1.0),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.2, 0.7),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
    )

    assert decision.source == "reactive"
    assert decision.command == DriveCommand(-0.2, 0.7)


def test_outputs_stop_when_both_commands_are_invalid():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(math.nan, 1.0),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(0.1, math.inf),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
    )

    assert decision.source == "stop"
    assert decision.command == DriveCommand(0.0, 0.0)


def test_selected_command_is_clamped():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(1.5, 9.0),
        mpc_health=healthy(0.95),
        reactive_command=None,
        reactive_health=SourceHealth(stamp_s=None, valid=False),
        odom_health=healthy(0.95),
        config=config(),
    )

    assert decision.source == "mpc"
    assert decision.command == DriveCommand(0.4, 2.0)


def test_shield_green_passes_mpc_when_margin_is_high():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.1, 1.2),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.3, 0.5),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(2.0, 0.8, 1.5),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
    )

    assert decision.source == "mpc"
    assert decision.command == DriveCommand(0.1, 1.2)
    assert "green" in decision.reason


def test_shield_yellow_limits_mpc_speed_and_keeps_steering():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.1, 1.4),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.3, 0.6),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(2.0, 0.5, 0.7),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
    )

    assert decision.source == "shield_yellow"
    assert decision.command == DriveCommand(0.1, 0.7)


def test_shield_orange_blends_steering_and_uses_conservative_speed():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.2, 1.3),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.4, 0.6),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(2.0, 0.25, 0.8),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
    )

    assert decision.source == "shield_orange"
    assert decision.command == DriveCommand(-0.1, 0.6)


def test_shield_red_uses_reactive_command():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.2, 1.3),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.4, 0.6),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(2.0, 0.1, 0.8),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
    )

    assert decision.source == "shield_red"
    assert decision.command == DriveCommand(-0.4, 0.6)


def test_shield_black_stops_when_front_clearance_is_too_low():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.2, 1.3),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.4, 0.6),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(0.2, 0.8, 0.5),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
    )

    assert decision.source == "stop"
    assert decision.command == DriveCommand(0.0, 0.0)
    assert "black" in decision.reason


def test_stale_shield_diagnostics_preserve_original_mpc_behavior():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.1, 1.2),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.3, 0.5),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(0.2, 0.1, 0.0),
        safety_health=healthy(0.0),
        shield_config=shield_config(),
    )

    assert decision.source == "mpc"
    assert decision.command == DriveCommand(0.1, 1.2)


def test_non_finite_shield_diagnostics_stop():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.1, 1.2),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.3, 0.5),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(math.nan, 0.8, 0.5),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
    )

    assert decision.source == "stop"
    assert "diagnostics" in decision.reason
