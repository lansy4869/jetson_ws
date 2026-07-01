from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class DriveCommand:
    steering_angle: float
    speed: float


@dataclass(frozen=True)
class SourceHealth:
    stamp_s: Optional[float]
    valid: bool
    reason: str = ""


@dataclass(frozen=True)
class ArbiterConfig:
    max_steer: float
    max_speed: float
    min_speed: float
    command_timeout_s: float
    odom_timeout_s: float
    prefer_reactive_on_mpc_timeout: bool


@dataclass(frozen=True)
class ArbitrationDecision:
    source: str
    command: DriveCommand
    reason: str


def _stop(reason: str) -> ArbitrationDecision:
    return ArbitrationDecision("stop", DriveCommand(0.0, 0.0), reason)


def _is_fresh(now_s: float, health: SourceHealth, timeout_s: float) -> bool:
    if not health.valid or health.stamp_s is None:
        return False
    if timeout_s < 0.0:
        return False
    return now_s - health.stamp_s <= timeout_s


def _fresh_reason(now_s: float, health: SourceHealth, timeout_s: float, name: str) -> str:
    if not health.valid:
        return health.reason or f"{name} invalid"
    if health.stamp_s is None:
        return f"{name} missing"
    if now_s - health.stamp_s > timeout_s:
        return f"{name} stale"
    return ""


def _is_command_finite(command: Optional[DriveCommand]) -> bool:
    if command is None:
        return False
    return math.isfinite(command.steering_angle) and math.isfinite(command.speed)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _clamp_command(command: DriveCommand, config: ArbiterConfig) -> DriveCommand:
    max_steer = abs(config.max_steer)
    min_speed = min(config.min_speed, config.max_speed)
    max_speed = max(config.min_speed, config.max_speed)
    return DriveCommand(
        steering_angle=_clamp(command.steering_angle, -max_steer, max_steer),
        speed=_clamp(command.speed, min_speed, max_speed),
    )


def arbitrate(
    now_s: float,
    mpc_command: Optional[DriveCommand],
    mpc_health: SourceHealth,
    reactive_command: Optional[DriveCommand],
    reactive_health: SourceHealth,
    odom_health: SourceHealth,
    config: ArbiterConfig,
) -> ArbitrationDecision:
    odom_reason = _fresh_reason(now_s, odom_health, config.odom_timeout_s, "odom")
    if odom_reason:
        return _stop(odom_reason)

    mpc_fresh = _is_fresh(now_s, mpc_health, config.command_timeout_s)
    if mpc_fresh and _is_command_finite(mpc_command):
        return ArbitrationDecision(
            "mpc",
            _clamp_command(mpc_command, config),
            "mpc healthy",
        )

    if config.prefer_reactive_on_mpc_timeout:
        reactive_fresh = _is_fresh(now_s, reactive_health, config.command_timeout_s)
        if reactive_fresh and _is_command_finite(reactive_command):
            return ArbitrationDecision(
                "reactive",
                _clamp_command(reactive_command, config),
                "reactive fallback",
            )

    mpc_reason = _fresh_reason(now_s, mpc_health, config.command_timeout_s, "mpc")
    if not mpc_reason and not _is_command_finite(mpc_command):
        mpc_reason = "mpc command invalid"
    reactive_reason = _fresh_reason(
        now_s,
        reactive_health,
        config.command_timeout_s,
        "reactive",
    )
    if not reactive_reason and not _is_command_finite(reactive_command):
        reactive_reason = "reactive command invalid"
    return _stop("; ".join(reason for reason in (mpc_reason, reactive_reason) if reason))
