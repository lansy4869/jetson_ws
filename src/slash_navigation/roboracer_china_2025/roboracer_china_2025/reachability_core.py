"""Mapless reachability scoring for reactive F1TENTH control.

The functions in this module intentionally avoid ROS dependencies so the
decision core can be unit-tested without a running ROS 2 environment.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ReachabilityConfig:
    wheelbase: float = 0.33
    vehicle_width: float = 0.32
    front_overhang: float = 0.36
    rear_overhang: float = 0.12
    max_steer: float = math.pi / 4.0
    steer_samples: int = 41
    horizon: float = 2.5
    path_step: float = 0.08
    max_speed: float = 2.5
    min_speed: float = 0.35
    max_brake_accel: float = 2.5
    system_delay: float = 0.16
    max_lateral_accel: float = 4.5
    base_margin: float = 0.16
    min_free_distance: float = 0.45
    range_min: float = 0.02
    range_max: float = 12.0
    fov_limit: float = math.pi / 2.0
    progress_weight: float = 2.4
    speed_weight: float = 0.8
    clearance_weight: float = 0.7
    steer_weight: float = 0.42
    steer_change_weight: float = 0.35
    unknown_weight: float = 2.0
    center_bias_weight: float = 0.08
    risk_speed_gain: float = 0.55
    risk_weight: float = 3.0
    confidence_weight: float = 1.0
    free_distance_risk_weight: float = 0.35
    low_clearance_risk_weight: float = 0.25
    unknown_risk_weight: float = 0.25
    dynamic_risk_weight: float = 0.65
    dynamic_closing_speed_threshold: float = 0.5
    dynamic_closing_speed_saturation: float = 2.5
    dynamic_risk_range: float = 8.0
    dynamic_max_delta_time: float = 0.5
    clearance_confidence_distance: float = 0.5
    confidence_floor: float = 0.05
    corridor_progress_weight: float = 1.4
    corridor_center_weight: float = 1.2
    corridor_heading_weight: float = 0.45
    corridor_confidence_speed_gain: float = 0.35
    corridor_single_boundary_speed_scale: float = 0.85
    corridor_unobservable_speed_scale: float = 0.55


@dataclass(frozen=True)
class ReachabilityResult:
    valid: bool
    steer: float
    speed: float
    free_distance: float
    min_clearance: float
    unknown_ratio: float
    score: float
    risk: float = 1.0
    confidence: float = 0.0
    dynamic_obstacle_risk: float = 0.0
    reason: str = ""


def select_reachable_gap(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    current_speed: float,
    last_steer: float,
    config: Optional[ReachabilityConfig] = None,
    previous_ranges: Optional[Iterable[float]] = None,
    delta_time: Optional[float] = None,
    corridor: Optional[object] = None,
) -> ReachabilityResult:
    """Select a steer/speed pair from reachable constant-curvature arcs."""

    cfg = config or ReachabilityConfig()
    scan = np.asarray(list(ranges), dtype=float)
    if scan.size == 0 or angle_increment == 0.0:
        return _invalid_result("empty_scan")

    angles = angle_min + np.arange(scan.size, dtype=float) * angle_increment
    valid = _valid_scan_mask(scan, angles, cfg)
    if not np.any(valid):
        return _invalid_result("no_valid_ranges")

    safe_ranges = np.where(valid, scan, cfg.range_max)
    obstacle_points = _scan_points(safe_ranges, angles, valid)
    dynamic_profile = _dynamic_risk_profile(
        scan=scan,
        angles=angles,
        valid=valid,
        previous_ranges=previous_ranges,
        delta_time=delta_time,
        cfg=cfg,
    )

    best = None
    for steer in np.linspace(-cfg.max_steer, cfg.max_steer, cfg.steer_samples):
        path = _rollout_arc(steer, cfg)
        free_distance, min_clearance, unknown_ratio, dynamic_obstacle_risk = _path_risk(
            path=path,
            scan_ranges=scan,
            scan_valid=valid,
            angle_min=angle_min,
            angle_increment=angle_increment,
            obstacle_points=obstacle_points,
            dynamic_profile=dynamic_profile,
            cfg=cfg,
        )

        if free_distance < cfg.min_free_distance:
            continue

        base_speed = _safe_speed(
            free_distance=free_distance,
            steer=steer,
            current_speed=current_speed,
            cfg=cfg,
        )
        risk = _candidate_risk(
            free_distance=free_distance,
            min_clearance=min_clearance,
            unknown_ratio=unknown_ratio,
            dynamic_obstacle_risk=dynamic_obstacle_risk,
            cfg=cfg,
        )
        confidence = _candidate_confidence(
            steer=steer,
            last_steer=last_steer,
            min_clearance=min_clearance,
            unknown_ratio=unknown_ratio,
            dynamic_obstacle_risk=dynamic_obstacle_risk,
            risk=risk,
            cfg=cfg,
        )
        speed = _risk_adjusted_speed(base_speed, risk, cfg)
        speed = _corridor_adjusted_speed(speed, corridor, cfg)
        score = _candidate_score(
            steer=steer,
            speed=speed,
            free_distance=free_distance,
            min_clearance=min_clearance,
            unknown_ratio=unknown_ratio,
            last_steer=last_steer,
            path=path,
            corridor=corridor,
            cfg=cfg,
        )
        score = score * (0.5 + 0.5 * confidence)
        score += cfg.confidence_weight * confidence
        score -= cfg.risk_weight * risk
        candidate = ReachabilityResult(
            valid=True,
            steer=float(steer),
            speed=float(speed),
            free_distance=float(free_distance),
            min_clearance=float(min_clearance),
            unknown_ratio=float(unknown_ratio),
            score=float(score),
            risk=float(risk),
            confidence=float(confidence),
            dynamic_obstacle_risk=float(dynamic_obstacle_risk),
            reason="ok",
        )
        if best is None or candidate.score > best.score:
            best = candidate

    if best is None:
        return _invalid_result("no_reachable_arc")
    return best


def _invalid_result(reason: str) -> ReachabilityResult:
    return ReachabilityResult(
        valid=False,
        steer=0.0,
        speed=0.0,
        free_distance=0.0,
        min_clearance=0.0,
        unknown_ratio=1.0,
        score=-float("inf"),
        risk=1.0,
        confidence=0.0,
        dynamic_obstacle_risk=0.0,
        reason=reason,
    )


def _valid_scan_mask(scan: np.ndarray, angles: np.ndarray, cfg: ReachabilityConfig) -> np.ndarray:
    finite = np.isfinite(scan)
    in_range = (scan > cfg.range_min) & (scan <= cfg.range_max)
    in_fov = np.abs(angles) <= cfg.fov_limit
    return finite & in_range & in_fov


def _scan_points(
    ranges: np.ndarray,
    angles: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    if not np.any(valid):
        return np.empty((0, 2), dtype=float)
    valid_ranges = ranges[valid]
    valid_angles = angles[valid]
    return np.column_stack(
        (
            valid_ranges * np.cos(valid_angles),
            valid_ranges * np.sin(valid_angles),
        )
    )


def _dynamic_risk_profile(
    scan: np.ndarray,
    angles: np.ndarray,
    valid: np.ndarray,
    previous_ranges: Optional[Iterable[float]],
    delta_time: Optional[float],
    cfg: ReachabilityConfig,
) -> np.ndarray:
    risk = np.zeros_like(scan, dtype=float)
    if (
        previous_ranges is None
        or delta_time is None
        or delta_time <= 1e-3
        or delta_time > cfg.dynamic_max_delta_time
    ):
        return risk

    previous = np.asarray(list(previous_ranges), dtype=float)
    if previous.size != scan.size:
        return risk

    previous_valid = np.isfinite(previous) & (previous > cfg.range_min) & (previous <= cfg.range_max)
    usable = valid & previous_valid & (np.abs(angles) <= cfg.fov_limit)
    if not np.any(usable):
        return risk

    closing_speed = (previous - scan) / max(delta_time, 1e-3)
    closing_risk = np.clip(
        (closing_speed - cfg.dynamic_closing_speed_threshold)
        / max(cfg.dynamic_closing_speed_saturation, 1e-6),
        0.0,
        1.0,
    )
    near_weight = np.clip(
        1.0 - scan / max(cfg.dynamic_risk_range, 1e-6),
        0.0,
        1.0,
    )
    risk[usable] = closing_risk[usable] * (0.35 + 0.65 * near_weight[usable])
    return risk


def _rollout_arc(steer: float, cfg: ReachabilityConfig) -> np.ndarray:
    samples = max(2, int(math.ceil(cfg.horizon / cfg.path_step)))
    s_values = np.linspace(cfg.path_step, cfg.horizon, samples)
    curvature = math.tan(steer) / max(cfg.wheelbase, 1e-6)

    if abs(curvature) < 1e-6:
        x = s_values
        y = np.zeros_like(s_values)
        yaw = np.zeros_like(s_values)
    else:
        x = np.sin(curvature * s_values) / curvature
        y = (1.0 - np.cos(curvature * s_values)) / curvature
        yaw = curvature * s_values

    return np.column_stack((s_values, x, y, yaw))


def _path_risk(
    path: np.ndarray,
    scan_ranges: np.ndarray,
    scan_valid: np.ndarray,
    angle_min: float,
    angle_increment: float,
    obstacle_points: np.ndarray,
    dynamic_profile: np.ndarray,
    cfg: ReachabilityConfig,
) -> Tuple[float, float, float, float]:
    min_clearance = float("inf")
    free_distance = cfg.horizon
    unknown_count = 0
    dynamic_obstacle_risk = 0.0

    for s_value, x, y, yaw in path:
        radial_clearance, is_unknown = _beam_clearance(
            x=x,
            y=y,
            scan_ranges=scan_ranges,
            scan_valid=scan_valid,
            angle_min=angle_min,
            angle_increment=angle_increment,
            cfg=cfg,
        )
        footprint_clearance = _footprint_clearance(
            x=x,
            y=y,
            yaw=yaw,
            obstacle_points=obstacle_points,
            cfg=cfg,
        )
        clearance = min(radial_clearance, footprint_clearance)
        min_clearance = min(min_clearance, clearance)

        if is_unknown:
            unknown_count += 1
        beam_index = _beam_index(x, y, angle_min, angle_increment)
        if 0 <= beam_index < dynamic_profile.size and scan_valid[beam_index]:
            dynamic_obstacle_risk = max(dynamic_obstacle_risk, float(dynamic_profile[beam_index]))
        if clearance <= 0.0:
            free_distance = float(s_value)
            break

    if not math.isfinite(min_clearance):
        min_clearance = cfg.range_max

    unknown_ratio = unknown_count / max(len(path), 1)
    return free_distance, min_clearance, unknown_ratio, dynamic_obstacle_risk


def _beam_index(
    x: float,
    y: float,
    angle_min: float,
    angle_increment: float,
) -> int:
    if angle_increment == 0.0:
        return -1
    angle = math.atan2(y, x)
    return int(round((angle - angle_min) / angle_increment))


def _beam_clearance(
    x: float,
    y: float,
    scan_ranges: np.ndarray,
    scan_valid: np.ndarray,
    angle_min: float,
    angle_increment: float,
    cfg: ReachabilityConfig,
) -> Tuple[float, bool]:
    index = _beam_index(x, y, angle_min, angle_increment)
    radial_distance = math.hypot(x, y) + cfg.front_overhang

    if index < 0 or index >= scan_ranges.size or not scan_valid[index]:
        return cfg.base_margin, True

    return float(scan_ranges[index] - radial_distance - cfg.base_margin), False


def _footprint_clearance(
    x: float,
    y: float,
    yaw: float,
    obstacle_points: np.ndarray,
    cfg: ReachabilityConfig,
) -> float:
    if obstacle_points.size == 0:
        return cfg.range_max

    dx = obstacle_points[:, 0] - x
    dy = obstacle_points[:, 1] - y
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy

    front = cfg.front_overhang + cfg.base_margin
    rear = cfg.rear_overhang + cfg.base_margin
    half_width = cfg.vehicle_width / 2.0 + cfg.base_margin

    outside_x = np.maximum.reduce((-rear - local_x, local_x - front, np.zeros_like(local_x)))
    outside_y = np.maximum(np.abs(local_y) - half_width, np.zeros_like(local_y))
    outside_distance = np.hypot(outside_x, outside_y)

    inside_x = (local_x >= -rear) & (local_x <= front)
    inside_y = np.abs(local_y) <= half_width
    signed = outside_distance
    if np.any(inside_x & inside_y):
        penetration = np.minimum.reduce(
            (
                local_x + rear,
                front - local_x,
                half_width - np.abs(local_y),
            )
        )
        signed = signed.copy()
        signed[inside_x & inside_y] = -penetration[inside_x & inside_y]

    return float(np.min(signed))


def _safe_speed(
    free_distance: float,
    steer: float,
    current_speed: float,
    cfg: ReachabilityConfig,
) -> float:
    usable_distance = max(0.0, free_distance - cfg.base_margin)
    brake = max(cfg.max_brake_accel, 1e-6)
    delay_term = brake * max(cfg.system_delay, 0.0)
    v_stop = -delay_term + math.sqrt(delay_term * delay_term + 2.0 * brake * usable_distance)

    curvature = abs(math.tan(steer) / max(cfg.wheelbase, 1e-6))
    if curvature < 1e-6:
        v_curve = cfg.max_speed
    else:
        v_curve = math.sqrt(max(cfg.max_lateral_accel, 0.0) / curvature)

    speed = min(cfg.max_speed, v_stop, v_curve)
    if current_speed <= cfg.min_speed and free_distance < cfg.min_free_distance * 1.5:
        return 0.0
    return float(np.clip(speed, cfg.min_speed, cfg.max_speed))


def _risk_adjusted_speed(speed: float, risk: float, cfg: ReachabilityConfig) -> float:
    factor = 1.0 - cfg.risk_speed_gain * np.clip(risk, 0.0, 1.0)
    return float(np.clip(speed * factor, 0.0, cfg.max_speed))


def _corridor_adjusted_speed(
    speed: float,
    corridor: Optional[object],
    cfg: ReachabilityConfig,
) -> float:
    if corridor is None:
        return speed

    confidence = _corridor_confidence(corridor)
    factor = 1.0 - cfg.corridor_confidence_speed_gain * (1.0 - confidence)
    mode = getattr(corridor, "observability_mode", "")
    if mode == "single_boundary":
        factor *= cfg.corridor_single_boundary_speed_scale
    elif mode == "unobservable":
        factor *= cfg.corridor_unobservable_speed_scale
    return float(np.clip(speed * factor, 0.0, cfg.max_speed))


def _candidate_risk(
    free_distance: float,
    min_clearance: float,
    unknown_ratio: float,
    dynamic_obstacle_risk: float,
    cfg: ReachabilityConfig,
) -> float:
    free_distance_risk = np.clip(
        1.0 - free_distance / max(cfg.horizon, 1e-6),
        0.0,
        1.0,
    )
    clearance_target = max(cfg.clearance_confidence_distance, cfg.base_margin)
    low_clearance_risk = np.clip(
        (clearance_target - min_clearance) / max(clearance_target, 1e-6),
        0.0,
        1.0,
    )
    risk = (
        cfg.free_distance_risk_weight * free_distance_risk
        + cfg.low_clearance_risk_weight * low_clearance_risk
        + cfg.unknown_risk_weight * np.clip(unknown_ratio, 0.0, 1.0)
        + cfg.dynamic_risk_weight * np.clip(dynamic_obstacle_risk, 0.0, 1.0)
    )
    return float(np.clip(risk, 0.0, 1.0))


def _candidate_confidence(
    steer: float,
    last_steer: float,
    min_clearance: float,
    unknown_ratio: float,
    dynamic_obstacle_risk: float,
    risk: float,
    cfg: ReachabilityConfig,
) -> float:
    clearance_confidence = np.clip(
        min_clearance / max(cfg.clearance_confidence_distance, 1e-6),
        0.0,
        1.0,
    )
    known_beam_confidence = 1.0 - np.clip(unknown_ratio, 0.0, 1.0)
    stability_confidence = 1.0 - 0.5 * np.clip(
        abs(steer - last_steer) / max(cfg.max_steer, 1e-6),
        0.0,
        1.0,
    )
    dynamic_confidence = 1.0 - np.clip(dynamic_obstacle_risk, 0.0, 1.0)
    risk_confidence = 1.0 - np.clip(risk, 0.0, 1.0)

    confidence = (
        clearance_confidence
        * known_beam_confidence
        * stability_confidence
        * dynamic_confidence
        * risk_confidence
    )
    confidence = cfg.confidence_floor + (1.0 - cfg.confidence_floor) * confidence
    return float(np.clip(confidence, 0.0, 1.0))


def _candidate_score(
    steer: float,
    speed: float,
    free_distance: float,
    min_clearance: float,
    unknown_ratio: float,
    last_steer: float,
    path: np.ndarray,
    corridor: Optional[object],
    cfg: ReachabilityConfig,
) -> float:
    bounded_clearance = max(min(min_clearance, cfg.range_max), -cfg.base_margin)
    base_score = (
        cfg.progress_weight * free_distance
        + cfg.speed_weight * speed
        + cfg.clearance_weight * bounded_clearance
        - cfg.steer_weight * abs(steer)
        - cfg.steer_change_weight * abs(steer - last_steer)
        - cfg.unknown_weight * unknown_ratio
        - cfg.center_bias_weight * abs(steer / max(cfg.max_steer, 1e-6))
    )
    return base_score + _corridor_score(path, corridor, cfg)


def _corridor_score(
    path: np.ndarray,
    corridor: Optional[object],
    cfg: ReachabilityConfig,
) -> float:
    if corridor is None or path.size == 0:
        return 0.0

    confidence = _corridor_confidence(corridor)
    if confidence <= 0.0:
        return 0.0

    x = float(path[-1, 1])
    y = float(path[-1, 2])
    yaw = float(path[-1, 3])
    tangent = float(getattr(corridor, "track_tangent", 0.0))
    center_offset = float(getattr(corridor, "center_offset", 0.0))
    estimated_width = float(getattr(corridor, "estimated_width", cfg.vehicle_width * 3.0))

    progress_along_tangent = x * math.cos(tangent) + y * math.sin(tangent)
    target_y = center_offset + x * math.tan(tangent)
    center_error = y - target_y
    heading_error = _angle_diff(yaw, tangent)
    width_scale = max(0.5 * estimated_width, cfg.vehicle_width, 1e-6)
    normalized_center_error = abs(center_error) / width_scale

    score = confidence * (
        cfg.corridor_progress_weight * progress_along_tangent
        - cfg.corridor_center_weight * normalized_center_error
        - cfg.corridor_heading_weight * abs(heading_error)
    )
    if getattr(corridor, "observability_mode", "") == "unobservable":
        score -= cfg.corridor_center_weight * (1.0 - confidence)
    return float(score)


def _corridor_confidence(corridor: object) -> float:
    return float(np.clip(getattr(corridor, "confidence", 0.0), 0.0, 1.0))


def _angle_diff(left: float, right: float) -> float:
    return math.atan2(math.sin(left - right), math.cos(left - right))
