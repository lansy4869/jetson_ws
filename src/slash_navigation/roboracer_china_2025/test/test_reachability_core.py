import math

import numpy as np

from roboracer_china_2025.reachability_core import (
    ReachabilityConfig,
    select_reachable_gap,
)


def make_scan(default_range=6.0, angle_min=-math.pi / 2, angle_max=math.pi / 2, count=181):
    angles = np.linspace(angle_min, angle_max, count)
    ranges = np.full(count, default_range, dtype=float)
    return angles, ranges


def block_sector(angles, ranges, start_deg, end_deg, distance):
    mask = (angles >= math.radians(start_deg)) & (angles <= math.radians(end_deg))
    ranges[mask] = distance


def test_open_scan_tracks_straight_with_high_speed():
    angles, ranges = make_scan(default_range=8.0)
    config = ReachabilityConfig(max_speed=3.0, horizon=2.5)

    result = select_reachable_gap(
        ranges=ranges,
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
        current_speed=1.0,
        last_steer=0.0,
        config=config,
    )

    assert result.valid
    assert abs(result.steer) < math.radians(3.0)
    assert result.speed > 2.5


def test_center_obstacle_prefers_reachable_open_right_side_and_slows_down():
    angles, ranges = make_scan(default_range=6.0)
    block_sector(angles, ranges, -10, 35, 1.2)
    config = ReachabilityConfig(
        max_speed=3.0,
        horizon=2.5,
        base_margin=0.18,
        unknown_weight=3.0,
    )

    result = select_reachable_gap(
        ranges=ranges,
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
        current_speed=1.2,
        last_steer=0.0,
        config=config,
    )

    assert result.valid
    assert result.steer < -math.radians(4.0)
    assert result.speed < 2.2
    assert result.min_clearance > 0.0
