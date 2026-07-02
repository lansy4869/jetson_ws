#ifndef FRENET_LOCAL_PLANNER_TRAJECTORY_GENERATOR_HPP_
#define FRENET_LOCAL_PLANNER_TRAJECTORY_GENERATOR_HPP_

#include "frenet_local_planner/state_machine.hpp"

#include <frenet_interfaces/msg/frenet_ego_state.hpp>
#include <frenet_interfaces/msg/frenet_trajectory_point.hpp>
#include <track_spline/frenet_converter.hpp>

#include <vector>

namespace frenet_local_planner
{

struct TrajectoryConfig
{
  int trajectory_points{20};
  double trajectory_ds{0.25};
  double lane_change_length_m{2.0};
};

class TrajectoryGenerator
{
public:
  explicit TrajectoryGenerator(TrajectoryConfig config);

  std::vector<frenet_interfaces::msg::FrenetTrajectoryPoint> generate(
    const frenet_interfaces::msg::FrenetEgoState & ego,
    const PlannerDecision & decision,
    const track_spline::FrenetConverter & converter) const;

private:
  double smoothstep(double value) const;
  double interpolateD(double ego_d, double target_d, double delta_s) const;

  TrajectoryConfig config_;
};

}  // namespace frenet_local_planner

#endif  // FRENET_LOCAL_PLANNER_TRAJECTORY_GENERATOR_HPP_
