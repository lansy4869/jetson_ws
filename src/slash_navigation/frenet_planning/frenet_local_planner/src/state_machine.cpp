#include "frenet_local_planner/state_machine.hpp"

#include <frenet_interfaces/msg/frenet_planner_state.hpp>

#include <cmath>

namespace frenet_local_planner
{

namespace
{

bool isFinite(double value)
{
  return std::isfinite(value);
}

bool isNonNegative(double value)
{
  return isFinite(value) && value >= 0.0;
}

bool isUnitInterval(double value)
{
  return isFinite(value) && value >= 0.0 && value <= 1.0;
}

bool hasFiniteAdvice(const frenet_interfaces::msg::ReactiveAdvice & advice)
{
  return
    isNonNegative(advice.front_clearance_m) &&
    isNonNegative(advice.left_clearance_m) &&
    isNonNegative(advice.right_clearance_m) &&
    isFinite(advice.best_gap_angle_rad) &&
    isNonNegative(advice.best_gap_width_rad) &&
    isUnitInterval(advice.best_gap_confidence) &&
    isUnitInterval(advice.dynamic_obstacle_confidence) &&
    isUnitInterval(advice.obstacle_risk);
}

bool hasFiniteEgo(const frenet_interfaces::msg::FrenetEgoState & ego)
{
  return
    isFinite(ego.s) &&
    isFinite(ego.s_unwrapped) &&
    isFinite(ego.d) &&
    isFinite(ego.yaw_error) &&
    isFinite(ego.speed);
}

}  // namespace

double stampToSeconds(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1.0e-9;
}

StateMachine::StateMachine(PlannerConfig config)
: config_(config)
{
}

PlannerDecision StateMachine::decide(
  const frenet_interfaces::msg::FrenetEgoState & ego,
  const frenet_interfaces::msg::ReactiveAdvice & advice,
  double now_seconds) const
{
  using frenet_interfaces::msg::FrenetPlannerState;

  if (
    !ego.valid ||
    ego.reinitialized ||
    !advice.valid ||
    !hasFiniteEgo(ego) ||
    !hasFiniteAdvice(advice) ||
    !isFresh(ego.header, now_seconds) ||
    !isFresh(advice.header, now_seconds))
  {
    return emergency();
  }

  if (
    advice.obstacle_risk >= config_.emergency_risk_threshold ||
    (advice.front_blocked && advice.front_clearance_m <= config_.emergency_front_clearance_m))
  {
    return emergency();
  }

  PlannerDecision decision;
  decision.valid = true;
  decision.state = FrenetPlannerState::GB_TRACK;
  decision.target_d = 0.0;
  decision.target_speed = config_.normal_speed;

  if (advice.front_blocked) {
    const bool confident = advice.best_gap_confidence >= config_.overtake_confidence_min;
    const bool left = advice.overtake_left_available;
    const bool right = advice.overtake_right_available;
    if (confident && (left || right)) {
      if (left && right) {
        constexpr double kGapAngleDeadbandRad = 1.0e-6;
        decision.state = advice.best_gap_angle_rad < -kGapAngleDeadbandRad ?
          FrenetPlannerState::OVERTAKE_RIGHT : FrenetPlannerState::OVERTAKE_LEFT;
      } else {
        decision.state = left ?
          FrenetPlannerState::OVERTAKE_LEFT : FrenetPlannerState::OVERTAKE_RIGHT;
      }
      decision.target_d = decision.state == FrenetPlannerState::OVERTAKE_LEFT ?
        config_.overtake_offset_m : -config_.overtake_offset_m;
      decision.target_speed = config_.overtake_speed;
      return decision;
    }

    decision.state = FrenetPlannerState::FOLLOW;
    decision.target_d = 0.0;
    decision.target_speed = config_.emergency_speed;
    return decision;
  }

  if (std::abs(ego.d) > config_.center_tolerance_m) {
    decision.state = FrenetPlannerState::RETURN_TO_CENTER;
    decision.target_d = 0.0;
    decision.target_speed = config_.normal_speed;
  }

  return decision;
}

bool StateMachine::isFresh(const std_msgs::msg::Header & header, double now_seconds) const
{
  if (!isFinite(now_seconds)) {
    return false;
  }
  if (header.stamp.sec == 0 && header.stamp.nanosec == 0U) {
    return false;
  }
  const double age = now_seconds - stampToSeconds(header.stamp);
  return age >= 0.0 && age <= config_.stale_timeout_s;
}

PlannerDecision StateMachine::emergency() const
{
  using frenet_interfaces::msg::FrenetPlannerState;
  PlannerDecision decision;
  decision.valid = false;
  decision.state = FrenetPlannerState::EMERGENCY;
  decision.target_d = 0.0;
  decision.target_speed = config_.emergency_speed;
  return decision;
}

}  // namespace frenet_local_planner
