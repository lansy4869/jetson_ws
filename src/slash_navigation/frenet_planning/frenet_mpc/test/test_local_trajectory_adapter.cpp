#include "frenet_mpc/local_trajectory_adapter.hpp"

#include <frenet_interfaces/msg/frenet_planner_state.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace
{

using frenet_interfaces::msg::FrenetPlannerState;

frenet_mpc::AdapterConfig makeConfig()
{
  frenet_mpc::AdapterConfig config;
  config.horizon = 4;
  config.dt = 0.5;
  config.min_sampling_speed = 0.2;
  config.min_trajectory_points = 5;
  return config;
}

frenet_interfaces::msg::FrenetLocalTrajectory makeStraightTrajectory(int points = 20)
{
  frenet_interfaces::msg::FrenetLocalTrajectory trajectory;
  trajectory.valid = true;
  trajectory.state = FrenetPlannerState::GB_TRACK;
  trajectory.target_speed = 1.0;
  trajectory.points.resize(static_cast<std::size_t>(points));

  for (int i = 0; i < points; ++i) {
    auto & point = trajectory.points[static_cast<std::size_t>(i)];
    point.s = static_cast<double>(i);
    point.d = 0.0;
    point.x = static_cast<double>(i);
    point.y = 0.0;
    point.yaw = 0.0;
    point.speed = 1.0;
  }

  return trajectory;
}

void expectRejected(const frenet_mpc::AdapterResult & result)
{
  EXPECT_FALSE(result.ok);
  EXPECT_FALSE(result.reason.empty());
  ASSERT_EQ(frenet_mpc::NX, result.ref_traj.rows());
  ASSERT_EQ(5, result.ref_traj.cols());
  EXPECT_TRUE(result.ref_traj.isZero(0.0));
}

}  // namespace

TEST(LocalTrajectoryAdapter, BuildsReferenceFromValidTrajectory)
{
  const frenet_mpc::LocalTrajectoryAdapter adapter(makeConfig());

  const auto result = adapter.buildReferenceTrajectory(
    makeStraightTrajectory(),
    frenet_mpc::Pose2D{0.0, 0.0, 0.0});

  ASSERT_TRUE(result.ok) << result.reason;
  EXPECT_TRUE(result.reason.empty());
  ASSERT_EQ(frenet_mpc::NX, result.ref_traj.rows());
  ASSERT_EQ(5, result.ref_traj.cols());

  EXPECT_NEAR(0.0, result.ref_traj(frenet_mpc::X, 0), 1.0e-9);
  EXPECT_NEAR(0.5, result.ref_traj(frenet_mpc::X, 1), 1.0e-9);
  EXPECT_NEAR(2.0, result.ref_traj(frenet_mpc::X, 4), 1.0e-9);
  EXPECT_NEAR(0.0, result.ref_traj(frenet_mpc::Y, 4), 1.0e-9);
  EXPECT_NEAR(1.0, result.ref_traj(frenet_mpc::V, 4), 1.0e-9);
  EXPECT_NEAR(0.0, result.ref_traj(frenet_mpc::YAW, 4), 1.0e-9);
}

TEST(LocalTrajectoryAdapter, StartsSamplingFromProjectedMiddle)
{
  const frenet_mpc::LocalTrajectoryAdapter adapter(makeConfig());

  const auto result = adapter.buildReferenceTrajectory(
    makeStraightTrajectory(),
    frenet_mpc::Pose2D{5.25, 0.4, 0.0});

  ASSERT_TRUE(result.ok) << result.reason;
  EXPECT_NEAR(5.25, result.ref_traj(frenet_mpc::X, 0), 1.0e-9);
  EXPECT_NEAR(0.0, result.ref_traj(frenet_mpc::Y, 0), 1.0e-9);
  EXPECT_NEAR(7.25, result.ref_traj(frenet_mpc::X, 4), 1.0e-9);
}

TEST(LocalTrajectoryAdapter, UnwrapsYawAcrossPiBoundary)
{
  auto trajectory = makeStraightTrajectory();
  trajectory.points[0].yaw = 3.10;
  trajectory.points[1].yaw = 3.13;
  trajectory.points[2].yaw = -3.12;
  trajectory.points[3].yaw = -3.10;

  const frenet_mpc::LocalTrajectoryAdapter adapter(makeConfig());
  const auto result = adapter.buildReferenceTrajectory(
    trajectory,
    frenet_mpc::Pose2D{0.0, 0.0, -3.13});

  ASSERT_TRUE(result.ok) << result.reason;
  EXPECT_LT(result.ref_traj(frenet_mpc::YAW, 0), -3.0);
  for (int i = 1; i < result.ref_traj.cols(); ++i) {
    EXPECT_LT(
      std::abs(result.ref_traj(frenet_mpc::YAW, i) -
        result.ref_traj(frenet_mpc::YAW, i - 1)),
      0.04);
  }
}

TEST(LocalTrajectoryAdapter, RejectsInvalidEmergencyAndShortTrajectory)
{
  const frenet_mpc::LocalTrajectoryAdapter adapter(makeConfig());

  auto invalid = makeStraightTrajectory();
  invalid.valid = false;
  expectRejected(adapter.buildReferenceTrajectory(invalid, frenet_mpc::Pose2D{}));

  auto emergency = makeStraightTrajectory();
  emergency.state = FrenetPlannerState::EMERGENCY;
  expectRejected(adapter.buildReferenceTrajectory(emergency, frenet_mpc::Pose2D{}));

  expectRejected(
    adapter.buildReferenceTrajectory(makeStraightTrajectory(4), frenet_mpc::Pose2D{}));
}

TEST(LocalTrajectoryAdapter, RejectsNonFiniteAndNegativeSpeed)
{
  const frenet_mpc::LocalTrajectoryAdapter adapter(makeConfig());

  auto nan_s = makeStraightTrajectory();
  nan_s.points[2].s = std::numeric_limits<double>::quiet_NaN();
  expectRejected(adapter.buildReferenceTrajectory(nan_s, frenet_mpc::Pose2D{}));

  auto inf_d = makeStraightTrajectory();
  inf_d.points[2].d = std::numeric_limits<double>::infinity();
  expectRejected(adapter.buildReferenceTrajectory(inf_d, frenet_mpc::Pose2D{}));

  auto nan_x = makeStraightTrajectory();
  nan_x.points[2].x = std::numeric_limits<double>::quiet_NaN();
  expectRejected(adapter.buildReferenceTrajectory(nan_x, frenet_mpc::Pose2D{}));

  auto negative_speed = makeStraightTrajectory();
  negative_speed.points[2].speed = -0.1;
  expectRejected(adapter.buildReferenceTrajectory(negative_speed, frenet_mpc::Pose2D{}));
}

TEST(LocalTrajectoryAdapter, RejectsWhenSamplingBeyondTrajectoryEnd)
{
  const frenet_mpc::LocalTrajectoryAdapter adapter(makeConfig());

  expectRejected(
    adapter.buildReferenceTrajectory(
      makeStraightTrajectory(5),
      frenet_mpc::Pose2D{3.0, 0.0, 0.0}));
}
