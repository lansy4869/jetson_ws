#ifndef FRENET_MPC_LOCAL_TRAJECTORY_ADAPTER_HPP_
#define FRENET_MPC_LOCAL_TRAJECTORY_ADAPTER_HPP_

#include "frenet_mpc/mpc_utils.hpp"

#include <frenet_interfaces/msg/frenet_local_trajectory.hpp>

#include <Eigen/Dense>

#include <string>

namespace frenet_mpc
{

struct Pose2D
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

struct AdapterConfig
{
  int horizon{12};
  double dt{0.1};
  double min_sampling_speed{0.2};
  int min_trajectory_points{13};
};

struct AdapterResult
{
  bool ok{false};
  std::string reason;
  Eigen::MatrixXd ref_traj;
};

class LocalTrajectoryAdapter
{
public:
  explicit LocalTrajectoryAdapter(AdapterConfig config);

  AdapterResult buildReferenceTrajectory(
    const frenet_interfaces::msg::FrenetLocalTrajectory & trajectory,
    const Pose2D & ego_pose) const;

private:
  AdapterConfig config_;
};

}  // namespace frenet_mpc

#endif  // FRENET_MPC_LOCAL_TRAJECTORY_ADAPTER_HPP_
