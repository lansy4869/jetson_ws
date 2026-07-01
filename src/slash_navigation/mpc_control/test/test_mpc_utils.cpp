#include "mpc_control/mpc_utils.hpp"

#include <gtest/gtest.h>

#include <OsqpEigen/Constants.hpp>

#include <Eigen/Dense>

#include <tuple>

namespace
{

TEST(MPCUtils, NearestPointUsesClosingSegment)
{
  Eigen::MatrixXd trajectory(2, 4);
  trajectory << 0.0, 10.0, 10.0, 0.0,
                0.0, 0.0, 10.0, 10.0;

  Eigen::Vector2d query(-0.2, 5.0);

  Eigen::Vector2d nearest;
  double distance = 0.0;
  double segment_t = 0.0;
  std::size_t segment_index = 0U;
  std::tie(nearest, distance, segment_t, segment_index) =
    mpc_control::nearestPoint(query, trajectory);

  EXPECT_EQ(3U, segment_index);
  EXPECT_NEAR(0.0, nearest.x(), 1.0e-9);
  EXPECT_NEAR(5.0, nearest.y(), 1.0e-9);
  EXPECT_NEAR(0.2, distance, 1.0e-9);
  EXPECT_NEAR(0.5, segment_t, 1.0e-9);
}

TEST(MPCUtils, ReachableSpeedBoundsAllowStandingStart)
{
  mpc_control::Config config;
  config.MIN_SPEED = 0.5;
  config.MAX_SPEED = 2.5;
  config.MAX_ACCEL = 2.0;
  config.DTK = 0.1;

  EXPECT_NEAR(0.0, mpc_control::reachableSpeedLowerBound(0.0, 0, config), 1.0e-12);
  EXPECT_NEAR(0.2, mpc_control::reachableSpeedLowerBound(0.0, 1, config), 1.0e-12);
  EXPECT_NEAR(0.4, mpc_control::reachableSpeedLowerBound(0.0, 2, config), 1.0e-12);
  EXPECT_NEAR(0.5, mpc_control::reachableSpeedLowerBound(0.0, 3, config), 1.0e-12);
}

TEST(MPCUtils, ReachableSpeedBoundsAllowOverspeedRecovery)
{
  mpc_control::Config config;
  config.MIN_SPEED = 0.5;
  config.MAX_SPEED = 2.5;
  config.MAX_ACCEL = 2.0;
  config.DTK = 0.1;

  EXPECT_NEAR(3.0, mpc_control::reachableSpeedUpperBound(3.0, 0, config), 1.0e-12);
  EXPECT_NEAR(2.8, mpc_control::reachableSpeedUpperBound(3.0, 1, config), 1.0e-12);
  EXPECT_NEAR(2.6, mpc_control::reachableSpeedUpperBound(3.0, 2, config), 1.0e-12);
  EXPECT_NEAR(2.5, mpc_control::reachableSpeedUpperBound(3.0, 3, config), 1.0e-12);
}

TEST(MPCUtils, CommandLowerBoundDoesNotForceMinSpeedFromRest)
{
  mpc_control::Config config;
  config.MIN_SPEED = 0.5;

  EXPECT_NEAR(0.0, mpc_control::commandSpeedLowerBound(0.0, config), 1.0e-12);
  EXPECT_NEAR(0.0, mpc_control::commandSpeedLowerBound(0.49, config), 1.0e-12);
  EXPECT_NEAR(0.5, mpc_control::commandSpeedLowerBound(0.5, config), 1.0e-12);
}

TEST(MPCUtils, OsqpStatusAcceptsOnlySolvedStates)
{
  EXPECT_TRUE(mpc_control::isOsqpSolvedStatus(OsqpEigen::Status::Solved));
  EXPECT_TRUE(mpc_control::isOsqpSolvedStatus(OsqpEigen::Status::SolvedInaccurate));
  EXPECT_FALSE(mpc_control::isOsqpSolvedStatus(OsqpEigen::Status::PrimalInfeasible));
  EXPECT_FALSE(mpc_control::isOsqpSolvedStatus(OsqpEigen::Status::DualInfeasible));
  EXPECT_FALSE(mpc_control::isOsqpSolvedStatus(OsqpEigen::Status::MaxIterReached));
  EXPECT_FALSE(mpc_control::isOsqpSolvedStatus(OsqpEigen::Status::Unsolved));
}

TEST(MPCUtils, UnwrapYawReferenceKeepsPiBoundaryContinuous)
{
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(mpc_control::NX, 4);
  ref(mpc_control::YAW, 0) = -3.12;
  ref(mpc_control::YAW, 1) = 3.13;
  ref(mpc_control::YAW, 2) = 3.10;
  ref(mpc_control::YAW, 3) = 3.05;

  mpc_control::unwrapYawReference(ref, -3.13);

  EXPECT_NEAR(-3.12, ref(mpc_control::YAW, 0), 1.0e-12);
  EXPECT_LT(std::abs(ref(mpc_control::YAW, 1) - ref(mpc_control::YAW, 0)), 0.04);
  EXPECT_LT(std::abs(ref(mpc_control::YAW, 2) - ref(mpc_control::YAW, 1)), 0.04);
  EXPECT_LT(std::abs(ref(mpc_control::YAW, 3) - ref(mpc_control::YAW, 2)), 0.06);
  EXPECT_LT(ref(mpc_control::YAW, 3), -3.0);
}

TEST(MPCUtils, UpdateStatePreservesYawContinuityAcrossNegativePi)
{
  mpc_control::Config config;
  config.DTK = 0.1;
  config.WB = 0.42;
  config.MIN_STEER = -0.412;
  config.MAX_STEER = 0.412;
  config.MIN_SPEED = 0.0;
  config.MAX_SPEED = 3.0;

  mpc_control::State state;
  state.v = 2.0;
  state.yaw = -3.13;

  const auto next = mpc_control::updateState(state, 0.0, -0.4, config);

  EXPECT_LT(next.yaw, -mpc_control::kPi);
  EXPECT_GT(next.yaw, -3.40);
}

}  // namespace
