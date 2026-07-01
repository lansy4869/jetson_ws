import math

from roboracer_china_2025.drive_arbiter_core import (
    ArbiterConfig,
    DriveCommand,
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
