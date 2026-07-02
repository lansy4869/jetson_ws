#ifndef FRENET_RUNTIME_EGO_FRENET_PROJECTOR_HPP_
#define FRENET_RUNTIME_EGO_FRENET_PROJECTOR_HPP_

#include "track_spline/frenet_converter.hpp"

namespace frenet_runtime
{

struct EgoFrenetProjectorConfig
{
  double near_search_radius_m{0.0};
  double max_near_projection_distance_m{0.0};
  double max_global_projection_distance_m{0.0};
  double max_projection_yaw_error_rad{1.5707963267948966};
};

struct ProjectedEgoState
{
  bool valid{false};
  bool reinitialized{false};
  double s{0.0};
  double s_unwrapped{0.0};
  double d{0.0};
  double yaw_error{0.0};
  double speed{0.0};
};

class EgoFrenetProjector
{
public:
  EgoFrenetProjector(
    track_spline::FrenetConverter converter,
    EgoFrenetProjectorConfig config);
  EgoFrenetProjector(const EgoFrenetProjector &) = delete;
  EgoFrenetProjector & operator=(const EgoFrenetProjector &) = delete;
  EgoFrenetProjector(EgoFrenetProjector &&) = default;
  EgoFrenetProjector & operator=(EgoFrenetProjector &&) = default;

  // Returns an all-zero invalid state on failure. On successful reinitialization,
  // s_unwrapped is reset to wrapped s and may therefore jump.
  ProjectedEgoState project(double x, double y, double yaw, double speed);
  // Clears projection history; the next valid projection reports reinitialized=true.
  void reset();

private:
  track_spline::FrenetConverter converter_;
  EgoFrenetProjectorConfig config_;
  bool has_history_{false};
  double previous_wrapped_s_{0.0};
  double previous_unwrapped_s_{0.0};
};

}  // namespace frenet_runtime

#endif  // FRENET_RUNTIME_EGO_FRENET_PROJECTOR_HPP_
