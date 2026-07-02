#include "frenet_local_planner/trajectory_generator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace frenet_local_planner
{

namespace
{

double finiteYawFromDelta(double dx, double dy, double fallback)
{
  if (std::hypot(dx, dy) <= 1.0e-9) {
    return fallback;
  }
  return std::atan2(dy, dx);
}

}  // namespace

TrajectoryGenerator::TrajectoryGenerator(TrajectoryConfig config)
: config_(config)
{
  if (config_.trajectory_points < 2) {
    throw std::invalid_argument("trajectory_points must be at least 2");
  }
  if (!std::isfinite(config_.trajectory_ds) || config_.trajectory_ds <= 0.0) {
    throw std::invalid_argument("trajectory_ds must be positive and finite");
  }
  if (!std::isfinite(config_.lane_change_length_m) || config_.lane_change_length_m <= 0.0) {
    throw std::invalid_argument("lane_change_length_m must be positive and finite");
  }
}

std::vector<frenet_interfaces::msg::FrenetTrajectoryPoint> TrajectoryGenerator::generate(
  const frenet_interfaces::msg::FrenetEgoState & ego,
  const PlannerDecision & decision,
  const track_spline::FrenetConverter & converter) const
{
  if (!ego.valid || !decision.valid || converter.empty()) {
    return {};
  }

  std::vector<frenet_interfaces::msg::FrenetTrajectoryPoint> points;
  points.reserve(static_cast<std::size_t>(config_.trajectory_points));

  for (int i = 0; i < config_.trajectory_points; ++i) {
    const double delta_s = static_cast<double>(i) * config_.trajectory_ds;
    const double s = ego.s_unwrapped + delta_s;
    const double d = interpolateD(ego.d, decision.target_d, delta_s);
    const auto xy = converter.frenetToXY(s, d);

    frenet_interfaces::msg::FrenetTrajectoryPoint point;
    point.s = s;
    point.d = d;
    point.x = xy.x;
    point.y = xy.y;
    point.yaw = xy.yaw;
    point.speed = decision.target_speed;
    points.push_back(point);
  }

  if (points.size() >= 2U) {
    for (std::size_t i = 0; i + 1U < points.size(); ++i) {
      points[i].yaw = finiteYawFromDelta(
        points[i + 1U].x - points[i].x,
        points[i + 1U].y - points[i].y,
        points[i].yaw);
    }
    points.back().yaw = points[points.size() - 2U].yaw;
  }

  return points;
}

double TrajectoryGenerator::smoothstep(double value) const
{
  const double t = std::max(0.0, std::min(1.0, value));
  return t * t * (3.0 - 2.0 * t);
}

double TrajectoryGenerator::interpolateD(double ego_d, double target_d, double delta_s) const
{
  return ego_d + (target_d - ego_d) * smoothstep(delta_s / config_.lane_change_length_m);
}

}  // namespace frenet_local_planner
