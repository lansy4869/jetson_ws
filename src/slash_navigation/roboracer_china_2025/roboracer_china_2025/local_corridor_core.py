"""Local track-corridor estimation for mapless F1TENTH racing.

This module has no ROS dependency.  It converts a single LaserScan-like range
array into a compact local corridor estimate that can condition the
reachability scorer without introducing a global map or global localization.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CorridorConfig:
    track_width_prior: float = 3.0
    min_track_width: float = 1.2
    max_track_width: float = 6.0
    range_min: float = 0.05
    range_max: float = 12.0
    boundary_max_range: float = 8.0
    side_min_angle: float = math.radians(28.0)
    side_max_angle: float = math.radians(88.0)
    tangent_lookahead: float = 4.0
    min_side_points: int = 5
    center_offset_limit: float = 1.5
    tangent_limit: float = math.radians(40.0)
    memory_decay: float = 0.65
    unobservable_confidence: float = 0.12


@dataclass(frozen=True)
class CorridorEstimate:
    observability_mode: str
    visible_boundary: str
    track_tangent: float
    center_offset: float
    estimated_width: float
    confidence: float


def estimate_local_corridor(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    previous: Optional[CorridorEstimate] = None,
    config: Optional[CorridorConfig] = None,
) -> CorridorEstimate:
    """Estimate local corridor center and tangent from side boundary returns."""

    cfg = config or CorridorConfig()
    scan = np.asarray(list(ranges), dtype=float)
    if scan.size == 0 or angle_increment == 0.0:
        return _unobservable(previous, cfg)

    angles = angle_min + np.arange(scan.size, dtype=float) * angle_increment
    finite = np.isfinite(scan)
    bounded = (scan > cfg.range_min) & (scan < min(cfg.range_max, cfg.boundary_max_range))
    side_fov = (np.abs(angles) >= cfg.side_min_angle) & (np.abs(angles) <= cfg.side_max_angle)
    valid = finite & bounded & side_fov
    if not np.any(valid):
        return _unobservable(previous, cfg)

    points = np.column_stack((scan[valid] * np.cos(angles[valid]), scan[valid] * np.sin(angles[valid])))
    point_angles = angles[valid]

    left = _side_boundary(points, point_angles, side="left", cfg=cfg)
    right = _side_boundary(points, point_angles, side="right", cfg=cfg)

    if left is not None and right is not None:
        left_distance, left_slope, left_count = left
        right_distance, right_slope, right_count = right
        width = float(np.clip(left_distance + right_distance, cfg.min_track_width, cfg.max_track_width))
        center_offset = _clip_center_offset((left_distance - right_distance) * 0.5, cfg)
        tangent = _clip_tangent(_mean_slope_angle(left_slope, right_slope), cfg)
        point_confidence = min(1.0, min(left_count, right_count) / max(float(cfg.min_side_points * 3), 1.0))
        width_confidence = _width_confidence(width, cfg)
        confidence = float(np.clip(0.55 + 0.35 * point_confidence + 0.10 * width_confidence, 0.0, 1.0))
        return CorridorEstimate(
            observability_mode="two_boundary",
            visible_boundary="both",
            track_tangent=tangent,
            center_offset=center_offset,
            estimated_width=width,
            confidence=confidence,
        )

    if left is not None:
        left_distance, left_slope, left_count = left
        width = _width_prior(cfg)
        center_offset = _clip_center_offset(left_distance - width * 0.5, cfg)
        confidence = _single_boundary_confidence(left_count, cfg)
        return CorridorEstimate(
            observability_mode="single_boundary",
            visible_boundary="left",
            track_tangent=_clip_tangent(math.atan(left_slope), cfg),
            center_offset=center_offset,
            estimated_width=width,
            confidence=confidence,
        )

    if right is not None:
        right_distance, right_slope, right_count = right
        width = _width_prior(cfg)
        center_offset = _clip_center_offset(width * 0.5 - right_distance, cfg)
        confidence = _single_boundary_confidence(right_count, cfg)
        return CorridorEstimate(
            observability_mode="single_boundary",
            visible_boundary="right",
            track_tangent=_clip_tangent(math.atan(right_slope), cfg),
            center_offset=center_offset,
            estimated_width=width,
            confidence=confidence,
        )

    return _unobservable(previous, cfg)


def _side_boundary(
    points: np.ndarray,
    angles: np.ndarray,
    side: str,
    cfg: CorridorConfig,
) -> Optional[Tuple[float, float, int]]:
    if side == "left":
        side_mask = angles > 0.0
        lateral = points[:, 1]
    else:
        side_mask = angles < 0.0
        lateral = -points[:, 1]

    forward = (points[:, 0] >= -0.05) & (points[:, 0] <= cfg.tangent_lookahead)
    mask = side_mask & forward & (lateral > cfg.range_min)
    if int(np.count_nonzero(mask)) < cfg.min_side_points:
        return None

    side_points = points[mask]
    distances = lateral[mask]
    distance = float(np.median(distances))
    slope = _fit_boundary_slope(side_points)
    return distance, slope, int(side_points.shape[0])


def _fit_boundary_slope(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    if points.shape[0] < 2 or float(np.max(x) - np.min(x)) < 0.2:
        return 0.0
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def _mean_slope_angle(left_slope: float, right_slope: float) -> float:
    return math.atan(0.5 * (left_slope + right_slope))


def _width_prior(cfg: CorridorConfig) -> float:
    return float(np.clip(cfg.track_width_prior, cfg.min_track_width, cfg.max_track_width))


def _width_confidence(width: float, cfg: CorridorConfig) -> float:
    prior = _width_prior(cfg)
    tolerance = max(0.5 * prior, 1e-6)
    return float(np.clip(1.0 - abs(width - prior) / tolerance, 0.0, 1.0))


def _single_boundary_confidence(point_count: int, cfg: CorridorConfig) -> float:
    point_confidence = min(1.0, point_count / max(float(cfg.min_side_points * 3), 1.0))
    return float(np.clip(0.28 + 0.22 * point_confidence, 0.0, 0.65))


def _unobservable(previous: Optional[CorridorEstimate], cfg: CorridorConfig) -> CorridorEstimate:
    if previous is None:
        tangent = 0.0
        center_offset = 0.0
        width = _width_prior(cfg)
    else:
        tangent = _clip_tangent(previous.track_tangent * cfg.memory_decay, cfg)
        center_offset = _clip_center_offset(previous.center_offset * cfg.memory_decay, cfg)
        width = float(np.clip(previous.estimated_width, cfg.min_track_width, cfg.max_track_width))

    return CorridorEstimate(
        observability_mode="unobservable",
        visible_boundary="none",
        track_tangent=tangent,
        center_offset=center_offset,
        estimated_width=width,
        confidence=float(np.clip(cfg.unobservable_confidence, 0.0, 0.3)),
    )


def _clip_tangent(value: float, cfg: CorridorConfig) -> float:
    return float(np.clip(value, -cfg.tangent_limit, cfg.tangent_limit))


def _clip_center_offset(value: float, cfg: CorridorConfig) -> float:
    return float(np.clip(value, -cfg.center_offset_limit, cfg.center_offset_limit))
