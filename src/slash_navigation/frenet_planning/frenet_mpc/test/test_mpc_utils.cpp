#include "frenet_mpc/mpc_utils.hpp"

#include <gtest/gtest.h>

#include <OsqpEigen/Constants.hpp>

#include <Eigen/Dense>

#include <cmath>

namespace
{

TEST(FrenetMPCUtils, ReachableSpeedBoundsAllowStandingStart)
{
  frenet_mpc::Config config;
  config.MIN_SPEED = 0.5;
  config.MAX_SPEED = 2.5;
  config.MAX_ACCEL = 2.0;
  config.DTK = 0.1;

  EXPECT_NEAR(0.0, frenet_mpc::reachableSpeedLowerBound(0.0, 0, config), 1.0e-12);
  EXPECT_NEAR(0.2, frenet_mpc::reachableSpeedLowerBound(0.0, 1, config), 1.0e-12);
  EXPECT_NEAR(0.4, frenet_mpc::reachableSpeedLowerBound(0.0, 2, config), 1.0e-12);
  EXPECT_NEAR(0.5, frenet_mpc::reachableSpeedLowerBound(0.0, 3, config), 1.0e-12);
}

TEST(FrenetMPCUtils, ReachableSpeedBoundsAllowOverspeedRecovery)
{
  frenet_mpc::Config config;
  config.MIN_SPEED = 0.5;
  config.MAX_SPEED = 2.5;
  config.MAX_ACCEL = 2.0;
  config.DTK = 0.1;

  EXPECT_NEAR(3.0, frenet_mpc::reachableSpeedUpperBound(3.0, 0, config), 1.0e-12);
  EXPECT_NEAR(2.8, frenet_mpc::reachableSpeedUpperBound(3.0, 1, config), 1.0e-12);
  EXPECT_NEAR(2.6, frenet_mpc::reachableSpeedUpperBound(3.0, 2, config), 1.0e-12);
  EXPECT_NEAR(2.5, frenet_mpc::reachableSpeedUpperBound(3.0, 3, config), 1.0e-12);
}

TEST(FrenetMPCUtils, CommandLowerBoundDoesNotForceMinSpeedFromRest)
{
  frenet_mpc::Config config;
  config.MIN_SPEED = 0.5;

  EXPECT_NEAR(0.0, frenet_mpc::commandSpeedLowerBound(0.0, config), 1.0e-12);
  EXPECT_NEAR(0.0, frenet_mpc::commandSpeedLowerBound(0.49, config), 1.0e-12);
  EXPECT_NEAR(0.5, frenet_mpc::commandSpeedLowerBound(0.5, config), 1.0e-12);
}

TEST(FrenetMPCUtils, OsqpStatusAcceptsOnlySolvedStates)
{
  EXPECT_TRUE(frenet_mpc::isOsqpSolvedStatus(OsqpEigen::Status::Solved));
  EXPECT_TRUE(frenet_mpc::isOsqpSolvedStatus(OsqpEigen::Status::SolvedInaccurate));
  EXPECT_FALSE(frenet_mpc::isOsqpSolvedStatus(OsqpEigen::Status::PrimalInfeasible));
  EXPECT_FALSE(frenet_mpc::isOsqpSolvedStatus(OsqpEigen::Status::DualInfeasible));
  EXPECT_FALSE(frenet_mpc::isOsqpSolvedStatus(OsqpEigen::Status::MaxIterReached));
  EXPECT_FALSE(frenet_mpc::isOsqpSolvedStatus(OsqpEigen::Status::Unsolved));
}

TEST(FrenetMPCUtils, UnwrapYawReferenceKeepsPiBoundaryContinuous)
{
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(frenet_mpc::NX, 4);
  ref(frenet_mpc::YAW, 0) = -3.12;
  ref(frenet_mpc::YAW, 1) = 3.13;
  ref(frenet_mpc::YAW, 2) = 3.10;
  ref(frenet_mpc::YAW, 3) = 3.05;

  frenet_mpc::unwrapYawReference(ref, -3.13);

  EXPECT_NEAR(-3.12, ref(frenet_mpc::YAW, 0), 1.0e-12);
  EXPECT_LT(std::abs(ref(frenet_mpc::YAW, 1) - ref(frenet_mpc::YAW, 0)), 0.04);
  EXPECT_LT(std::abs(ref(frenet_mpc::YAW, 2) - ref(frenet_mpc::YAW, 1)), 0.04);
  EXPECT_LT(std::abs(ref(frenet_mpc::YAW, 3) - ref(frenet_mpc::YAW, 2)), 0.06);
  EXPECT_LT(ref(frenet_mpc::YAW, 3), -3.0);
}

TEST(FrenetMPCUtils, UpdateStatePreservesYawContinuityAcrossNegativePi)
{
  frenet_mpc::Config config;
  config.DTK = 0.1;
  config.WB = 0.42;
  config.MIN_STEER = -0.412;
  config.MAX_STEER = 0.412;
  config.MIN_SPEED = 0.0;
  config.MAX_SPEED = 3.0;

  frenet_mpc::State state;
  state.v = 2.0;
  state.yaw = -3.13;

  const auto next = frenet_mpc::updateState(state, 0.0, -0.4, config);

  EXPECT_LT(next.yaw, -frenet_mpc::kPi);
  EXPECT_GT(next.yaw, -3.40);
}

}  // namespace
