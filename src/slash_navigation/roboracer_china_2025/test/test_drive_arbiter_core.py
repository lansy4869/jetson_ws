import math

from roboracer_china_2025.drive_arbiter_core import (
    ArbiterConfig,
    DriveCommand,
    SafetyDiagnostics,
    ShieldConfig,
    SourceHealth,
    TrackConsistencyConfig,
    TrackConsistencyDiagnostics,
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


def track_config():
    return TrackConsistencyConfig(
        enabled=True,
        diag_timeout_s=0.2,
        yellow_lateral_error_m=0.35,
        orange_lateral_error_m=0.70,
        red_lateral_error_m=1.20,
        yellow_yaw_error_rad=0.35,
        orange_yaw_error_rad=0.70,
        red_yaw_error_rad=1.20,
        yellow_speed_limit_mps=0.9,
        orange_speed_limit_mps=0.6,
    )


def track_state(
    valid=True,
    reinitialized=False,
    lateral_error_m=0.05,
    yaw_error_rad=0.05,
    speed_mps=1.0,
):
    return TrackConsistencyDiagnostics(
        valid=valid,
        reinitialized=reinitialized,
        lateral_error_m=lateral_error_m,
        yaw_error_rad=yaw_error_rad,
        speed_mps=speed_mps,
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


def test_track_yellow_limits_mpc_speed_when_reachability_is_green():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.1, 1.4),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.3, 0.6),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(2.0, 0.8, 2.0),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
        track_diagnostics=track_state(lateral_error_m=0.5),
        track_health=healthy(0.95),
        track_config=track_config(),
    )

    assert decision.source == "track_yellow"
    assert decision.command == DriveCommand(0.1, 0.9)
    assert "track yellow" in decision.reason


def test_track_orange_blends_with_reactive_and_conservative_speed():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.2, 1.4),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.4, 0.8),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(2.0, 0.8, 2.0),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
        track_diagnostics=track_state(lateral_error_m=0.85),
        track_health=healthy(0.95),
        track_config=track_config(),
    )

    assert decision.source == "track_orange"
    assert decision.command == DriveCommand(-0.1, 0.6)
    assert "track orange" in decision.reason


def test_track_red_stops_even_when_reactive_command_is_available():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.2, 1.4),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.4, 0.8),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(2.0, 0.8, 2.0),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
        track_diagnostics=track_state(lateral_error_m=1.4),
        track_health=healthy(0.95),
        track_config=track_config(),
    )

    assert decision.source == "stop"
    assert decision.command == DriveCommand(0.0, 0.0)
    assert "track red" in decision.reason


def test_invalid_frenet_projection_stops_when_track_diagnostics_are_fresh():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.2, 1.4),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.4, 0.8),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(2.0, 0.8, 2.0),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
        track_diagnostics=track_state(valid=False),
        track_health=healthy(0.95),
        track_config=track_config(),
    )

    assert decision.source == "stop"
    assert decision.command == DriveCommand(0.0, 0.0)
    assert "track black" in decision.reason


def test_stale_track_diagnostics_preserve_existing_shield_behavior():
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
        track_diagnostics=track_state(lateral_error_m=1.4),
        track_health=healthy(0.0),
        track_config=track_config(),
    )

    assert decision.source == "mpc"
    assert decision.command == DriveCommand(0.1, 1.2)
