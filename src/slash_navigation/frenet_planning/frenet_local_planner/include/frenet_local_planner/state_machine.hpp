#ifndef FRENET_LOCAL_PLANNER_STATE_MACHINE_HPP_
#define FRENET_LOCAL_PLANNER_STATE_MACHINE_HPP_

#include <frenet_interfaces/msg/frenet_ego_state.hpp>
#include <frenet_interfaces/msg/reactive_advice.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <std_msgs/msg/header.hpp>

#include <cstdint>

namespace frenet_local_planner
{

struct PlannerConfig
{
  double overtake_offset_m{0.45};
  double center_tolerance_m{0.10};
  double stale_timeout_s{0.5};
  double normal_speed{2.0};
  double follow_speed{0.8};
  double overtake_speed{1.5};
  double emergency_speed{0.0};
  double overtake_confidence_min{0.6};
  double emergency_risk_threshold{0.9};
  double emergency_front_clearance_m{0.30};
};

struct PlannerDecision
{
  bool valid{false};
  uint8_t state{0U};
  double target_d{0.0};
  double target_speed{0.0};
};

class StateMachine
{
public:
  explicit StateMachine(PlannerConfig config);

  PlannerDecision decide(
    const frenet_interfaces::msg::FrenetEgoState & ego,
    const frenet_interfaces::msg::ReactiveAdvice & advice,
    double now_seconds) const;

private:
  bool isFresh(const std_msgs::msg::Header & header, double now_seconds) const;
  PlannerDecision emergency() const;

  PlannerConfig config_;
};

double stampToSeconds(const builtin_interfaces::msg::Time & stamp);

}  // namespace frenet_local_planner

#endif  // FRENET_LOCAL_PLANNER_STATE_MACHINE_HPP_
