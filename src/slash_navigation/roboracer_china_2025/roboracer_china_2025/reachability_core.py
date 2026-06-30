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
    vehicle_width: float = 0.29
    front_overhang: float = 0.20
    rear_overhang: float = 0.35
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


@dataclass(frozen=True)
class ReachabilityResult:
    valid: bool
    steer: float
    speed: float
    free_distance: float
    min_clearance: float
    unknown_ratio: float
    score: float
    reason: str = ""


def select_reachable_gap(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    current_speed: float,
    last_steer: float,
    config: Optional[ReachabilityConfig] = None,
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

    best = None
    for steer in np.linspace(-cfg.max_steer, cfg.max_steer, cfg.steer_samples):
        path = _rollout_arc(steer, cfg)
        free_distance, min_clearance, unknown_ratio = _path_risk(
            path=path,
            scan_ranges=scan,
            scan_valid=valid,
            angle_min=angle_min,
            angle_increment=angle_increment,
            obstacle_points=obstacle_points,
            cfg=cfg,
        )

        if free_distance < cfg.min_free_distance:
            continue

        speed = _safe_speed(
            free_distance=free_distance,
            steer=steer,
            current_speed=current_speed,
            cfg=cfg,
        )
        score = _candidate_score(
            steer=steer,
            speed=speed,
            free_distance=free_distance,
            min_clearance=min_clearance,
            unknown_ratio=unknown_ratio,
            last_steer=last_steer,
            cfg=cfg,
        )
        candidate = ReachabilityResult(
            valid=True,
            steer=float(steer),
            speed=float(speed),
            free_distance=float(free_distance),
            min_clearance=float(min_clearance),
            unknown_ratio=float(unknown_ratio),
            score=float(score),
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
    cfg: ReachabilityConfig,
) -> Tuple[float, float, float]:
    min_clearance = float("inf")
    free_distance = cfg.horizon
    unknown_count = 0

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
        if clearance <= 0.0:
            free_distance = float(s_value)
            break

    if not math.isfinite(min_clearance):
        min_clearance = cfg.range_max

    unknown_ratio = unknown_count / max(len(path), 1)
    return free_distance, min_clearance, unknown_ratio


def _beam_clearance(
    x: float,
    y: float,
    scan_ranges: np.ndarray,
    scan_valid: np.ndarray,
    angle_min: float,
    angle_increment: float,
    cfg: ReachabilityConfig,
) -> Tuple[float, bool]:
    angle = math.atan2(y, x)
    index = int(round((angle - angle_min) / angle_increment))
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


def _candidate_score(
    steer: float,
    speed: float,
    free_distance: float,
    min_clearance: float,
    unknown_ratio: float,
    last_steer: float,
    cfg: ReachabilityConfig,
) -> float:
    bounded_clearance = max(min(min_clearance, cfg.range_max), -cfg.base_margin)
    return (
        cfg.progress_weight * free_distance
        + cfg.speed_weight * speed
        + cfg.clearance_weight * bounded_clearance
        - cfg.steer_weight * abs(steer)
        - cfg.steer_change_weight * abs(steer - last_steer)
        - cfg.unknown_weight * unknown_ratio
        - cfg.center_bias_weight * abs(steer / max(cfg.max_steer, 1e-6))
    )
