import math

import numpy as np

from roboracer_china_2025.local_corridor_core import (
    CorridorConfig,
    CorridorEstimate,
    estimate_local_corridor,
)


def make_corridor_scan(
    left_distance=None,
    right_distance=None,
    angle_min=-math.pi / 2,
    angle_max=math.pi / 2,
    count=181,
    range_max=12.0,
):
    angles = np.linspace(angle_min, angle_max, count)
    ranges = np.full(count, range_max, dtype=float)
    for index, angle in enumerate(angles):
        candidates = []
        if left_distance is not None and math.sin(angle) > 1e-6:
            candidates.append(left_distance / math.sin(angle))
        if right_distance is not None and math.sin(angle) < -1e-6:
            candidates.append(-right_distance / math.sin(angle))
        candidates = [value for value in candidates if 0.05 < value <= range_max]
        if candidates:
            ranges[index] = min(candidates)
    return angles, ranges


def test_two_boundary_corridor_estimates_center_offset_and_width():
    angles, ranges = make_corridor_scan(left_distance=1.8, right_distance=1.2)

    corridor = estimate_local_corridor(
        ranges=ranges,
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
        config=CorridorConfig(track_width_prior=3.0),
    )

    assert corridor.observability_mode == "two_boundary"
    assert corridor.visible_boundary == "both"
    assert corridor.estimated_width == pytest_approx(3.0, abs=0.2)
    assert corridor.center_offset == pytest_approx(0.3, abs=0.08)
    assert corridor.confidence > 0.65


def test_single_boundary_corridor_uses_width_prior_for_virtual_center():
    angles, ranges = make_corridor_scan(left_distance=1.0, right_distance=None)

    corridor = estimate_local_corridor(
        ranges=ranges,
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
        config=CorridorConfig(track_width_prior=3.0),
    )

    assert corridor.observability_mode == "single_boundary"
    assert corridor.visible_boundary == "left"
    assert corridor.estimated_width == pytest_approx(3.0, abs=0.01)
    assert corridor.center_offset == pytest_approx(-0.5, abs=0.1)
    assert 0.25 <= corridor.confidence <= 0.65


def test_unobservable_corridor_decays_previous_direction_and_slows_confidence():
    angles, ranges = make_corridor_scan(left_distance=None, right_distance=None)
    previous = CorridorEstimate(
        observability_mode="two_boundary",
        visible_boundary="both",
        track_tangent=0.2,
        center_offset=-0.4,
        estimated_width=3.2,
        confidence=0.8,
    )

    corridor = estimate_local_corridor(
        ranges=ranges,
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
        previous=previous,
        config=CorridorConfig(track_width_prior=3.0, memory_decay=0.5),
    )

    assert corridor.observability_mode == "unobservable"
    assert corridor.visible_boundary == "none"
    assert corridor.track_tangent == pytest_approx(0.1, abs=1e-6)
    assert corridor.center_offset == pytest_approx(-0.2, abs=1e-6)
    assert corridor.confidence < 0.25


def pytest_approx(*args, **kwargs):
    import pytest

    return pytest.approx(*args, **kwargs)
