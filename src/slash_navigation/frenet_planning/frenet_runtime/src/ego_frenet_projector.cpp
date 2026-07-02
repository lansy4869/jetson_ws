#include "frenet_runtime/ego_frenet_projector.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace
{

constexpr double kPi = 3.14159265358979323846;

bool isFinite(double value)
{
  return std::isfinite(value);
}

double projectionDistance(
  double x,
  double y,
  const track_spline::FrenetPoint & point)
{
  return std::hypot(x - point.reference_x, y - point.reference_y);
}

void validateConfig(const frenet_runtime::EgoFrenetProjectorConfig & config)
{
  if (
    !isFinite(config.near_search_radius_m) ||
    !isFinite(config.max_near_projection_distance_m) ||
    !isFinite(config.max_global_projection_distance_m) ||
    !isFinite(config.max_projection_yaw_error_rad))
  {
    throw std::invalid_argument("ego_frenet_projector: config values must be finite");
  }

  if (config.near_search_radius_m <= 0.0) {
    throw std::invalid_argument("ego_frenet_projector: near_search_radius_m must be positive");
  }
  if (config.max_near_projection_distance_m <= 0.0) {
    throw std::invalid_argument(
            "ego_frenet_projector: max_near_projection_distance_m must be positive");
  }
  if (config.max_global_projection_distance_m <= 0.0) {
    throw std::invalid_argument(
            "ego_frenet_projector: max_global_projection_distance_m must be positive");
  }
  if (config.max_global_projection_distance_m < config.max_near_projection_distance_m) {
    throw std::invalid_argument(
            "ego_frenet_projector: max_global_projection_distance_m must be >= max_near_projection_distance_m");
  }
  if (config.max_projection_yaw_error_rad <= 0.0 || config.max_projection_yaw_error_rad > kPi) {
    throw std::invalid_argument(
            "ego_frenet_projector: max_projection_yaw_error_rad must be in (0, pi]");
  }
}

frenet_runtime::ProjectedEgoState makeValidState(
  const track_spline::FrenetPoint & point,
  double s_unwrapped,
  double speed,
  bool reinitialized)
{
  frenet_runtime::ProjectedEgoState state;
  state.valid = true;
  state.reinitialized = reinitialized;
  state.s = point.s;
  state.s_unwrapped = s_unwrapped;
  state.d = point.d;
  state.yaw_error = point.yaw_error;
  state.speed = speed;
  return state;
}

}  // namespace

namespace frenet_runtime
{

EgoFrenetProjector::EgoFrenetProjector(
  track_spline::FrenetConverter converter,
  EgoFrenetProjectorConfig config)
: converter_(std::move(converter)), config_(config)
{
  if (converter_.empty()) {
    throw std::invalid_argument("ego_frenet_projector: converter must be built");
  }
  validateConfig(config_);
}

ProjectedEgoState EgoFrenetProjector::project(double x, double y, double yaw, double speed)
{
  if (!isFinite(x) || !isFinite(y) || !isFinite(yaw) || !isFinite(speed)) {
    reset();
    return ProjectedEgoState{};
  }

  try {
    if (!has_history_) {
      const auto global_point = converter_.xyToFrenet(x, y, yaw);
      if (
        projectionDistance(x, y, global_point) > config_.max_global_projection_distance_m ||
        std::abs(global_point.yaw_error) > config_.max_projection_yaw_error_rad)
      {
        reset();
        return ProjectedEgoState{};
      }

      const auto state = makeValidState(global_point, global_point.s, speed, true);
      has_history_ = true;
      previous_wrapped_s_ = state.s;
      previous_unwrapped_s_ = state.s_unwrapped;
      return state;
    }

    const auto near_point = converter_.xyToFrenetNear(
      x,
      y,
      yaw,
      previous_wrapped_s_,
      config_.near_search_radius_m);
    if (
      projectionDistance(x, y, near_point) <= config_.max_near_projection_distance_m &&
      std::abs(near_point.yaw_error) <= config_.max_projection_yaw_error_rad)
    {
      const auto state = makeValidState(
        near_point,
        converter_.unwrapS(previous_unwrapped_s_, near_point.s),
        speed,
        false);
      has_history_ = true;
      previous_wrapped_s_ = state.s;
      previous_unwrapped_s_ = state.s_unwrapped;
      return state;
    }

    const auto global_point = converter_.xyToFrenet(x, y, yaw);
    if (
      projectionDistance(x, y, global_point) <= config_.max_global_projection_distance_m &&
      std::abs(global_point.yaw_error) <= config_.max_projection_yaw_error_rad)
    {
      const auto state = makeValidState(global_point, global_point.s, speed, true);
      has_history_ = true;
      previous_wrapped_s_ = state.s;
      previous_unwrapped_s_ = state.s_unwrapped;
      return state;
    }
  } catch (const std::exception &) {
    reset();
    return ProjectedEgoState{};
  }

  reset();
  return ProjectedEgoState{};
}

void EgoFrenetProjector::reset()
{
  has_history_ = false;
  previous_wrapped_s_ = 0.0;
  previous_unwrapped_s_ = 0.0;
}

}  // namespace frenet_runtime
