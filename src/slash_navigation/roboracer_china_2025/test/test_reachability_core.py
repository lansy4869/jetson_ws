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


def test_approaching_scan_reduces_speed_and_confidence():
    angles, ranges = make_scan(default_range=4.0)
    previous_ranges = np.full_like(ranges, 7.0)
    config = ReachabilityConfig(
        max_speed=3.0,
        horizon=2.5,
        risk_speed_gain=0.7,
        dynamic_risk_weight=1.2,
    )

    static_result = select_reachable_gap(
        ranges=ranges,
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
        current_speed=1.0,
        last_steer=0.0,
        previous_ranges=ranges,
        delta_time=0.2,
        config=config,
    )
    dynamic_result = select_reachable_gap(
        ranges=ranges,
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
        current_speed=1.0,
        last_steer=0.0,
        previous_ranges=previous_ranges,
        delta_time=0.2,
        config=config,
    )

    assert static_result.valid
    assert dynamic_result.valid
    assert dynamic_result.dynamic_obstacle_risk > 0.5
    assert dynamic_result.risk > static_result.risk
    assert dynamic_result.confidence < static_result.confidence
    assert dynamic_result.speed < static_result.speed


def test_near_low_clearance_scan_has_lower_confidence_than_open_scan():
    angles, open_ranges = make_scan(default_range=8.0)
    _, near_ranges = make_scan(default_range=1.8)
    config = ReachabilityConfig(max_speed=3.0, horizon=2.5)

    open_result = select_reachable_gap(
        ranges=open_ranges,
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
        current_speed=1.0,
        last_steer=0.0,
        config=config,
    )
    near_result = select_reachable_gap(
        ranges=near_ranges,
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
        current_speed=1.0,
        last_steer=0.0,
        config=config,
    )

    assert open_result.valid
    assert near_result.valid
    assert near_result.risk > open_result.risk
    assert near_result.confidence < open_result.confidence
    assert near_result.speed < open_result.speed


def test_dynamic_risk_can_shift_choice_away_from_approaching_side():
    angles, ranges = make_scan(default_range=3.0)
    block_sector(angles, ranges, -10, 10, 2.2)
    block_sector(angles, ranges, 15, 65, 8.0)
    block_sector(angles, ranges, -65, -15, 3.0)

    previous_ranges = ranges.copy()
    left_mask = (angles >= math.radians(15)) & (angles <= math.radians(65))
    previous_ranges[left_mask] = 10.0

    config = ReachabilityConfig(
        max_speed=3.0,
        horizon=2.5,
        dynamic_risk_weight=1.4,
        risk_weight=5.0,
        confidence_weight=1.5,
    )

    static_result = select_reachable_gap(
        ranges=ranges,
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
        current_speed=1.0,
        last_steer=0.0,
        previous_ranges=ranges,
        delta_time=0.2,
        config=config,
    )
    dynamic_result = select_reachable_gap(
        ranges=ranges,
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
        current_speed=1.0,
        last_steer=0.0,
        previous_ranges=previous_ranges,
        delta_time=0.2,
        config=config,
    )

    assert static_result.valid
    assert dynamic_result.valid
    assert static_result.steer > math.radians(4.0)
    assert dynamic_result.steer < -math.radians(4.0)
