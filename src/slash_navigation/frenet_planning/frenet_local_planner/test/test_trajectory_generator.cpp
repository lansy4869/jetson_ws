#include "frenet_local_planner/trajectory_generator.hpp"

#include <frenet_interfaces/msg/frenet_planner_state.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace
{

using frenet_interfaces::msg::FrenetEgoState;
using frenet_interfaces::msg::FrenetPlannerState;

constexpr double kPi = 3.14159265358979323846;

track_spline::FrenetConverter makeSquareConverter()
{
  track_spline::FrenetConverter converter;
  converter.build(
    std::vector<track_spline::TrackPoint>{
      track_spline::TrackPoint{0.0, 0.0, 0.0},
      track_spline::TrackPoint{4.0, 0.0, 0.0},
      track_spline::TrackPoint{4.0, 4.0, 0.5 * kPi},
      track_spline::TrackPoint{0.0, 4.0, kPi}});
  return converter;
}

FrenetEgoState makeEgo()
{
  FrenetEgoState ego;
  ego.valid = true;
  ego.s = 1.0;
  ego.s_unwrapped = 1.0;
  ego.d = 0.0;
  ego.speed = 1.0;
  return ego;
}

frenet_local_planner::TrajectoryConfig makeConfig()
{
  frenet_local_planner::TrajectoryConfig config;
  config.trajectory_points = 8;
  config.trajectory_ds = 0.5;
  config.lane_change_length_m = 2.0;
  return config;
}

frenet_local_planner::PlannerDecision makeDecision(uint8_t state, double target_d)
{
  frenet_local_planner::PlannerDecision decision;
  decision.valid = true;
  decision.state = state;
  decision.target_d = target_d;
  decision.target_speed = 1.5;
  return decision;
}

}  // namespace

TEST(FrenetTrajectoryGeneratorTest, GlobalTrackingStaysNearCenterline)
{
  frenet_local_planner::TrajectoryGenerator generator(makeConfig());
  const auto points = generator.generate(
    makeEgo(),
    makeDecision(FrenetPlannerState::GB_TRACK, 0.0),
    makeSquareConverter());

  ASSERT_EQ(8U, points.size());
  EXPECT_NEAR(0.0, points.front().d, 1.0e-9);
  EXPECT_NEAR(0.0, points.back().d, 1.0e-9);
  EXPECT_NEAR(1.0, points.front().s, 1.0e-9);
  EXPECT_TRUE(std::isfinite(points.front().x));
  EXPECT_TRUE(std::isfinite(points.front().yaw));
}

TEST(FrenetTrajectoryGeneratorTest, OvertakeLeftGeneratesPositiveOffset)
{
  frenet_local_planner::TrajectoryGenerator generator(makeConfig());
  const auto points = generator.generate(
    makeEgo(),
    makeDecision(FrenetPlannerState::OVERTAKE_LEFT, 0.45),
    makeSquareConverter());

  ASSERT_EQ(8U, points.size());
  EXPECT_NEAR(0.0, points.front().d, 1.0e-9);
  EXPECT_GT(points.back().d, 0.40);
  EXPECT_NEAR(1.5, points.back().speed, 1.0e-9);
}

TEST(FrenetTrajectoryGeneratorTest, ReturnToCenterSmoothlyReducesOffset)
{
  auto ego = makeEgo();
  ego.d = 0.5;
  frenet_local_planner::TrajectoryGenerator generator(makeConfig());
  const auto points = generator.generate(
    ego,
    makeDecision(FrenetPlannerState::RETURN_TO_CENTER, 0.0),
    makeSquareConverter());

  ASSERT_EQ(8U, points.size());
  EXPECT_NEAR(0.5, points.front().d, 1.0e-9);
  EXPECT_LT(points.back().d, 0.1);
  for (std::size_t i = 1; i < points.size(); ++i) {
    EXPECT_LE(points[i].d, points[i - 1].d + 1.0e-9);
  }
}

TEST(FrenetTrajectoryGeneratorTest, WraparoundArcLengthRemainsFinite)
{
  auto converter = makeSquareConverter();
  auto ego = makeEgo();
  ego.s_unwrapped = converter.length() - 0.2;
  frenet_local_planner::TrajectoryGenerator generator(makeConfig());

  const auto points = generator.generate(
    ego,
    makeDecision(FrenetPlannerState::GB_TRACK, 0.0),
    converter);

  ASSERT_EQ(8U, points.size());
  for (const auto & point : points) {
    EXPECT_TRUE(std::isfinite(point.x));
    EXPECT_TRUE(std::isfinite(point.y));
    EXPECT_TRUE(std::isfinite(point.yaw));
  }
}

TEST(FrenetTrajectoryGeneratorTest, RejectsSinglePointTrajectory)
{
  auto config = makeConfig();
  config.trajectory_points = 1;
  EXPECT_THROW(
    frenet_local_planner::TrajectoryGenerator generator(config),
    std::invalid_argument);
}
