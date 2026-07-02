#include "frenet_mpc/local_trajectory_adapter.hpp"

#include <frenet_interfaces/msg/frenet_planner_state.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace frenet_mpc
{
namespace
{

struct PreparedPoint
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
  double speed{0.0};
  double distance{0.0};
};

struct InterpolatedSample
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
  double speed{0.0};
};

Eigen::MatrixXd zeroReference(int horizon)
{
  return Eigen::MatrixXd::Zero(NX, horizon + 1);
}

AdapterResult reject(const std::string & reason, int horizon)
{
  AdapterResult result;
  result.ok = false;
  result.reason = reason;
  result.ref_traj = zeroReference(horizon);
  return result;
}

bool isFinitePose(const Pose2D & pose)
{
  return std::isfinite(pose.x) && std::isfinite(pose.y) && std::isfinite(pose.yaw);
}

bool isFinitePoint(
  const frenet_interfaces::msg::FrenetTrajectoryPoint & point)
{
  return std::isfinite(point.s) && std::isfinite(point.d) &&
         std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.yaw) && std::isfinite(point.speed);
}

bool preparePoints(
  const frenet_interfaces::msg::FrenetLocalTrajectory & trajectory,
  std::vector<PreparedPoint> & points,
  std::string & reason)
{
  points.clear();
  points.reserve(trajectory.points.size());

  for (const auto & input : trajectory.points) {
    if (!isFinitePoint(input)) {
      reason = "trajectory contains non-finite point";
      return false;
    }
    if (input.speed < 0.0) {
      reason = "trajectory contains negative speed";
      return false;
    }

    PreparedPoint point;
    point.x = input.x;
    point.y = input.y;
    point.speed = input.speed;
    point.yaw = points.empty() ? input.yaw : unwrapAngleNear(input.yaw, points.back().yaw);

    if (!points.empty()) {
      const double dx = point.x - points.back().x;
      const double dy = point.y - points.back().y;
      point.distance = points.back().distance + std::hypot(dx, dy);
    }

    points.push_back(point);
  }

  if (points.empty() || points.back().distance <= kNearZero) {
    reason = "trajectory polyline has zero length";
    return false;
  }

  return true;
}

bool projectOntoPolyline(
  const std::vector<PreparedPoint> & points,
  const Pose2D & ego_pose,
  double & projected_distance)
{
  double best_distance_sq = std::numeric_limits<double>::infinity();
  bool found_projection = false;

  for (std::size_t i = 1; i < points.size(); ++i) {
    const auto & start = points[i - 1];
    const auto & end = points[i];
    const double dx = end.x - start.x;
    const double dy = end.y - start.y;
    const double length_sq = dx * dx + dy * dy;
    const double length = std::sqrt(length_sq);
    if (length <= kNearZero) {
      continue;
    }

    const double t = clampValue(
      ((ego_pose.x - start.x) * dx + (ego_pose.y - start.y) * dy) / length_sq,
      0.0, 1.0);
    const double projected_x = start.x + t * dx;
    const double projected_y = start.y + t * dy;
    const double error_x = ego_pose.x - projected_x;
    const double error_y = ego_pose.y - projected_y;
    const double distance_sq = error_x * error_x + error_y * error_y;

    if (distance_sq < best_distance_sq) {
      best_distance_sq = distance_sq;
      projected_distance = start.distance + t * length;
      found_projection = true;
    }
  }

  return found_projection;
}

bool interpolateAtDistance(
  const std::vector<PreparedPoint> & points,
  double distance,
  InterpolatedSample & sample)
{
  if (points.empty() || distance < -kNearZero) {
    return false;
  }

  if (distance <= kNearZero) {
    sample.x = points.front().x;
    sample.y = points.front().y;
    sample.yaw = points.front().yaw;
    sample.speed = points.front().speed;
    return true;
  }

  const double end_distance = points.back().distance;
  if (distance > end_distance + kNearZero) {
    return false;
  }

  for (std::size_t i = 1; i < points.size(); ++i) {
    const auto & start = points[i - 1];
    const auto & end = points[i];
    const double span = end.distance - start.distance;
    if (span <= kNearZero) {
      continue;
    }
    if (distance <= end.distance + kNearZero) {
      const double t = clampValue((distance - start.distance) / span, 0.0, 1.0);
      sample.x = start.x + t * (end.x - start.x);
      sample.y = start.y + t * (end.y - start.y);
      sample.yaw = start.yaw + t * (end.yaw - start.yaw);
      sample.speed = start.speed + t * (end.speed - start.speed);
      return true;
    }
  }

  sample.x = points.back().x;
  sample.y = points.back().y;
  sample.yaw = points.back().yaw;
  sample.speed = points.back().speed;
  return true;
}

}  // namespace

LocalTrajectoryAdapter::LocalTrajectoryAdapter(AdapterConfig config)
: config_(config)
{
  if (config_.horizon < 1) {
    throw std::invalid_argument("adapter horizon must be >= 1");
  }
  if (!std::isfinite(config_.dt) || config_.dt <= 0.0) {
    throw std::invalid_argument("adapter dt must be finite and > 0");
  }
  if (!std::isfinite(config_.min_sampling_speed) || config_.min_sampling_speed < 0.0) {
    throw std::invalid_argument("adapter min_sampling_speed must be finite and >= 0");
  }
  if (config_.min_trajectory_points < config_.horizon + 1) {
    throw std::invalid_argument("adapter min_trajectory_points must be >= horizon + 1");
  }
}

AdapterResult LocalTrajectoryAdapter::buildReferenceTrajectory(
  const frenet_interfaces::msg::FrenetLocalTrajectory & trajectory,
  const Pose2D & ego_pose) const
{
  if (!trajectory.valid) {
    return reject("trajectory is invalid", config_.horizon);
  }
  if (trajectory.state == frenet_interfaces::msg::FrenetPlannerState::EMERGENCY) {
    return reject("trajectory is in emergency state", config_.horizon);
  }
  if (trajectory.points.size() < static_cast<std::size_t>(config_.min_trajectory_points)) {
    return reject("trajectory has too few points", config_.horizon);
  }
  if (!isFinitePose(ego_pose)) {
    return reject("ego pose is non-finite", config_.horizon);
  }

  std::vector<PreparedPoint> points;
  std::string reason;
  if (!preparePoints(trajectory, points, reason)) {
    return reject(reason, config_.horizon);
  }

  double sample_distance = 0.0;
  if (!projectOntoPolyline(points, ego_pose, sample_distance)) {
    return reject("ego projection onto trajectory failed", config_.horizon);
  }

  Eigen::MatrixXd ref_traj = zeroReference(config_.horizon);
  const double trajectory_end = points.back().distance;
  for (int i = 0; i <= config_.horizon; ++i) {
    if (sample_distance > trajectory_end + kNearZero) {
      return reject("sample distance exceeds trajectory end", config_.horizon);
    }

    InterpolatedSample sample;
    if (!interpolateAtDistance(points, sample_distance, sample)) {
      return reject("failed to sample trajectory", config_.horizon);
    }

    ref_traj(X, i) = sample.x;
    ref_traj(Y, i) = sample.y;
    ref_traj(V, i) = sample.speed;
    ref_traj(YAW, i) = sample.yaw;

    sample_distance += std::max(sample.speed, config_.min_sampling_speed) * config_.dt;
  }

  unwrapYawReference(ref_traj, ego_pose.yaw);

  AdapterResult result;
  result.ok = true;
  result.ref_traj = ref_traj;
  return result;
}

}  // namespace frenet_mpc
