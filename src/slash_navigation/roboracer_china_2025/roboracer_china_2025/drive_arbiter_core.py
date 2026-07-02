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
class SafetyDiagnostics:
    front_clearance: float
    risk_min_margin: float
    reactive_speed_limit: float


@dataclass(frozen=True)
class ShieldConfig:
    enabled: bool
    diag_timeout_s: float
    yellow_margin_m: float
    orange_margin_m: float
    red_margin_m: float
    black_clearance_m: float
    orange_blend: float


@dataclass(frozen=True)
class TrackConsistencyDiagnostics:
    valid: bool
    reinitialized: bool
    lateral_error_m: float
    yaw_error_rad: float
    speed_mps: float


@dataclass(frozen=True)
class TrackConsistencyConfig:
    enabled: bool
    diag_timeout_s: float
    yellow_lateral_error_m: float
    orange_lateral_error_m: float
    red_lateral_error_m: float
    yellow_yaw_error_rad: float
    orange_yaw_error_rad: float
    red_yaw_error_rad: float
    yellow_speed_limit_mps: float
    orange_speed_limit_mps: float


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


def _is_diagnostics_finite(
    diagnostics: Optional[SafetyDiagnostics],
) -> bool:
    if diagnostics is None:
        return False
    return (
        math.isfinite(diagnostics.front_clearance)
        and math.isfinite(diagnostics.risk_min_margin)
        and math.isfinite(diagnostics.reactive_speed_limit)
    )


def _is_track_diagnostics_finite(
    diagnostics: Optional[TrackConsistencyDiagnostics],
) -> bool:
    if diagnostics is None:
        return False
    return (
        math.isfinite(diagnostics.lateral_error_m)
        and math.isfinite(diagnostics.yaw_error_rad)
        and math.isfinite(diagnostics.speed_mps)
    )


def _shield_active(
    now_s: float,
    safety_health: Optional[SourceHealth],
    shield_config: Optional[ShieldConfig],
) -> bool:
    if shield_config is None or not shield_config.enabled:
        return False
    if safety_health is None:
        return False
    return _is_fresh(now_s, safety_health, shield_config.diag_timeout_s)


def _track_consistency_active(
    now_s: float,
    track_health: Optional[SourceHealth],
    track_config: Optional[TrackConsistencyConfig],
) -> bool:
    if track_config is None or not track_config.enabled:
        return False
    if track_health is None:
        return False
    return _is_fresh(now_s, track_health, track_config.diag_timeout_s)


def _risk_state(
    diagnostics: SafetyDiagnostics,
    shield_config: ShieldConfig,
) -> str:
    if not _is_diagnostics_finite(diagnostics):
        return "black"
    if diagnostics.reactive_speed_limit < 0.0:
        return "black"
    if diagnostics.front_clearance <= shield_config.black_clearance_m:
        return "black"
    if diagnostics.risk_min_margin <= shield_config.red_margin_m:
        return "red"
    if diagnostics.risk_min_margin <= shield_config.orange_margin_m:
        return "orange"
    if diagnostics.risk_min_margin <= shield_config.yellow_margin_m:
        return "yellow"
    return "green"


def _track_consistency_state(
    diagnostics: TrackConsistencyDiagnostics,
    track_config: TrackConsistencyConfig,
) -> str:
    if not _is_track_diagnostics_finite(diagnostics):
        return "black"
    if not diagnostics.valid:
        return "black"

    lateral_error = abs(diagnostics.lateral_error_m)
    yaw_error = abs(diagnostics.yaw_error_rad)
    if (
        lateral_error >= track_config.red_lateral_error_m
        or yaw_error >= track_config.red_yaw_error_rad
    ):
        return "red"
    if (
        lateral_error >= track_config.orange_lateral_error_m
        or yaw_error >= track_config.orange_yaw_error_rad
    ):
        return "orange"
    if (
        diagnostics.reinitialized
        or lateral_error >= track_config.yellow_lateral_error_m
        or yaw_error >= track_config.yellow_yaw_error_rad
    ):
        return "yellow"
    return "green"


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
    safety_diagnostics: Optional[SafetyDiagnostics] = None,
    safety_health: Optional[SourceHealth] = None,
    shield_config: Optional[ShieldConfig] = None,
    track_diagnostics: Optional[TrackConsistencyDiagnostics] = None,
    track_health: Optional[SourceHealth] = None,
    track_config: Optional[TrackConsistencyConfig] = None,
) -> ArbitrationDecision:
    odom_reason = _fresh_reason(now_s, odom_health, config.odom_timeout_s, "odom")
    if odom_reason:
        return _stop(odom_reason)

    mpc_fresh = _is_fresh(now_s, mpc_health, config.command_timeout_s)
    reactive_fresh = _is_fresh(now_s, reactive_health, config.command_timeout_s)

    local_shield_active = _shield_active(now_s, safety_health, shield_config)
    track_shield_active = _track_consistency_active(
        now_s,
        track_health,
        track_config,
    )

    if local_shield_active or track_shield_active:
        state = "green"
        local_speed_limit = config.max_speed
        if local_shield_active:
            if not _is_diagnostics_finite(safety_diagnostics):
                return _stop("shield diagnostics non-finite")

            state = _risk_state(safety_diagnostics, shield_config)
            local_speed_limit = safety_diagnostics.reactive_speed_limit
            if state == "black":
                return _stop("shield black")

        track_state = "green"
        track_speed_limit = config.max_speed
        if track_shield_active:
            if not _is_track_diagnostics_finite(track_diagnostics):
                return _stop("track diagnostics non-finite")

            track_state = _track_consistency_state(track_diagnostics, track_config)
            if track_state == "black":
                return _stop("track black")
            if track_state == "red":
                return _stop("track red")
            if track_state == "orange":
                track_speed_limit = track_config.orange_speed_limit_mps
            elif track_state == "yellow":
                track_speed_limit = track_config.yellow_speed_limit_mps

        combined_speed_limit = min(local_speed_limit, track_speed_limit)

        if state == "red":
            if reactive_fresh and _is_command_finite(reactive_command):
                return ArbitrationDecision(
                    "shield_red",
                    _clamp_command(reactive_command, config),
                    "shield red reactive",
                )
            return _stop("shield red reactive unavailable")

        if mpc_fresh and _is_command_finite(mpc_command):
            if state == "orange" or track_state == "orange":
                if reactive_fresh and _is_command_finite(reactive_command):
                    orange_blend = 0.5
                    if shield_config is not None:
                        orange_blend = shield_config.orange_blend
                    blend = _clamp(orange_blend, 0.0, 1.0)
                    blended = DriveCommand(
                        (1.0 - blend) * mpc_command.steering_angle
                        + blend * reactive_command.steering_angle,
                        min(
                            mpc_command.speed,
                            reactive_command.speed,
                            combined_speed_limit,
                        ),
                    )
                    if track_state == "orange" and state != "orange":
                        return ArbitrationDecision(
                            "track_orange",
                            _clamp_command(blended, config),
                            "track orange blend",
                        )
                    return ArbitrationDecision(
                        "shield_orange",
                        _clamp_command(blended, config),
                        "shield orange blend",
                    )

                limited = DriveCommand(
                    mpc_command.steering_angle,
                    min(mpc_command.speed, combined_speed_limit),
                )
                if track_state == "orange" and state != "orange":
                    return ArbitrationDecision(
                        "track_yellow",
                        _clamp_command(limited, config),
                        "track orange degraded to speed limit",
                    )
                return ArbitrationDecision(
                    "shield_yellow",
                    _clamp_command(limited, config),
                    "shield orange degraded to speed limit",
                )

            if (
                state == "yellow"
                or track_state == "yellow"
                or (
                    local_shield_active
                    and state == "green"
                    and mpc_command.speed > local_speed_limit
                )
            ):
                limited = DriveCommand(
                    mpc_command.steering_angle,
                    min(mpc_command.speed, combined_speed_limit),
                )
                if track_state == "yellow" and state != "yellow":
                    return ArbitrationDecision(
                        "track_yellow",
                        _clamp_command(limited, config),
                        "track yellow speed limit",
                    )
                if state == "green":
                    return ArbitrationDecision(
                        "shield_yellow",
                        _clamp_command(limited, config),
                        "shield green speed limited",
                    )
                return ArbitrationDecision(
                    "shield_yellow",
                    _clamp_command(limited, config),
                    "shield yellow speed limit",
                )

            return ArbitrationDecision(
                "mpc",
                _clamp_command(mpc_command, config),
                "shield green",
            )

    if mpc_fresh and _is_command_finite(mpc_command):
        return ArbitrationDecision(
            "mpc",
            _clamp_command(mpc_command, config),
            "mpc healthy",
        )

    if config.prefer_reactive_on_mpc_timeout:
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
