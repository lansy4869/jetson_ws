#include "frenet_runtime/ego_frenet_projector.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

constexpr double kPi = 3.14159265358979323846;

std::vector<track_spline::TrackPoint> makeSquareLoopPoints()
{
  return std::vector<track_spline::TrackPoint>{
    track_spline::TrackPoint{0.0, 0.0, 0.0},
    track_spline::TrackPoint{4.0, 0.0, 0.0},
    track_spline::TrackPoint{4.0, 4.0, 0.5 * kPi},
    track_spline::TrackPoint{0.0, 4.0, kPi}};
}

track_spline::FrenetConverter makeSquareConverter()
{
  track_spline::FrenetConverter converter;
  converter.build(makeSquareLoopPoints());
  return converter;
}

track_spline::FrenetConverter makePathologicalConverter()
{
  // Regression case that exercises the projector's exception-to-invalid recovery path.
  track_spline::FrenetConverter converter;
  converter.build(
    std::vector<track_spline::TrackPoint>{
      track_spline::TrackPoint{9.02111, 7.56096, -2.26822},
      track_spline::TrackPoint{-7.44992, -5.40885, -0.134923},
      track_spline::TrackPoint{3.7286, 3.83465, -1.24989},
      track_spline::TrackPoint{-4.25229, -2.78675, 2.08495}});
  return converter;
}

}  // namespace

TEST(EgoFrenetProjectorTest, InitializesGloballyThenUsesNearProjection)
{
  auto converter = makeSquareConverter();
  frenet_runtime::EgoFrenetProjector projector(
    converter,
    frenet_runtime::EgoFrenetProjectorConfig{1.0, 0.3, 0.5});

  const auto first_xy = converter.frenetToXY(1.0, 0.05);
  const auto second_xy = converter.frenetToXY(1.3, 0.05);

  const auto first = projector.project(first_xy.x, first_xy.y, first_xy.yaw, 2.5);
  const auto second = projector.project(second_xy.x, second_xy.y, second_xy.yaw, 3.0);

  EXPECT_TRUE(first.valid);
  EXPECT_TRUE(first.reinitialized);
  EXPECT_NEAR(1.0, first.s, 1.0e-6);
  EXPECT_NEAR(1.0, first.s_unwrapped, 1.0e-6);
  EXPECT_NEAR(0.05, first.d, 1.0e-6);
  EXPECT_NEAR(0.0, first.yaw_error, 1.0e-6);
  EXPECT_DOUBLE_EQ(2.5, first.speed);

  EXPECT_TRUE(second.valid);
  EXPECT_FALSE(second.reinitialized);
  EXPECT_NEAR(1.3, second.s, 1.0e-6);
  EXPECT_GT(second.s_unwrapped, first.s_unwrapped);
  EXPECT_NEAR(0.3, second.s_unwrapped - first.s_unwrapped, 1.0e-6);
  EXPECT_DOUBLE_EQ(3.0, second.speed);
}

TEST(EgoFrenetProjectorTest, UnwrapsAcrossClosedLoopBoundary)
{
  auto converter = makeSquareConverter();
  frenet_runtime::EgoFrenetProjector projector(
    converter,
    frenet_runtime::EgoFrenetProjectorConfig{0.6, 0.2, 0.4});

  const double length = converter.length();
  const auto before_xy = converter.frenetToXY(length - 0.1, 0.0);
  const auto after_xy = converter.frenetToXY(0.1, 0.0);

  const auto before = projector.project(before_xy.x, before_xy.y, before_xy.yaw, 1.0);
  const auto after = projector.project(after_xy.x, after_xy.y, after_xy.yaw, 1.0);

  EXPECT_TRUE(before.valid);
  EXPECT_TRUE(after.valid);
  EXPECT_FALSE(after.reinitialized);
  EXPECT_GT(before.s, after.s);
  EXPECT_NEAR(0.2, after.s_unwrapped - before.s_unwrapped, 1.0e-6);
}

TEST(EgoFrenetProjectorTest, RelocalizesThenInvalidatesAndRecovers)
{
  auto converter = makeSquareConverter();
  frenet_runtime::EgoFrenetProjector projector(
    converter,
    frenet_runtime::EgoFrenetProjectorConfig{0.25, 0.2, 0.6});

  const auto first_xy = converter.frenetToXY(0.2, 0.0);
  const auto far_xy = converter.frenetToXY(3.2, 0.0);
  const auto recover_xy = converter.frenetToXY(0.6, 0.0);

  const auto first = projector.project(first_xy.x, first_xy.y, first_xy.yaw, 1.5);
  projector.reset();
  const auto after_reset = projector.project(first_xy.x, first_xy.y, first_xy.yaw, 1.55);
  const auto relocalized = projector.project(far_xy.x, far_xy.y, far_xy.yaw, 1.6);
  const auto invalid_far = projector.project(100.0, 100.0, 0.0, 1.7);
  const auto invalid_nan = projector.project(
    std::numeric_limits<double>::quiet_NaN(),
    0.0,
    0.0,
    1.8);
  const auto recovered = projector.project(recover_xy.x, recover_xy.y, recover_xy.yaw, 1.9);

  EXPECT_TRUE(first.valid);
  EXPECT_TRUE(after_reset.valid);
  EXPECT_TRUE(after_reset.reinitialized);
  EXPECT_NEAR(0.2, after_reset.s, 1.0e-6);
  EXPECT_TRUE(relocalized.valid);
  EXPECT_TRUE(relocalized.reinitialized);
  EXPECT_NEAR(3.2, relocalized.s, 1.0e-6);

  EXPECT_FALSE(invalid_far.valid);
  EXPECT_FALSE(invalid_far.reinitialized);
  EXPECT_DOUBLE_EQ(0.0, invalid_far.speed);

  EXPECT_FALSE(invalid_nan.valid);
  EXPECT_FALSE(invalid_nan.reinitialized);
  EXPECT_DOUBLE_EQ(0.0, invalid_nan.speed);

  EXPECT_TRUE(recovered.valid);
  EXPECT_TRUE(recovered.reinitialized);
  EXPECT_NEAR(0.6, recovered.s, 1.0e-6);

  frenet_runtime::EgoFrenetProjector pathological_projector(
    makePathologicalConverter(),
    frenet_runtime::EgoFrenetProjectorConfig{1.0, 1.0, 1.0});
  const auto pathological = pathological_projector.project(9.08452, 7.48363, -2.45477, 1.0);

  EXPECT_FALSE(pathological.valid);
  EXPECT_FALSE(pathological.reinitialized);
  EXPECT_DOUBLE_EQ(0.0, pathological.s);
  EXPECT_DOUBLE_EQ(0.0, pathological.s_unwrapped);
  EXPECT_DOUBLE_EQ(0.0, pathological.d);
  EXPECT_DOUBLE_EQ(0.0, pathological.yaw_error);
  EXPECT_DOUBLE_EQ(0.0, pathological.speed);
}

TEST(EgoFrenetProjectorTest, RejectsProjectionWithLargeYawError)
{
  auto converter = makeSquareConverter();
  frenet_runtime::EgoFrenetProjector projector(
    converter,
    frenet_runtime::EgoFrenetProjectorConfig{1.0, 0.3, 0.5, 1.0});

  const auto pose = converter.frenetToXY(1.0, 0.0);

  const auto projected = projector.project(pose.x, pose.y, pose.yaw + kPi, 1.0);

  EXPECT_FALSE(projected.valid);
  EXPECT_FALSE(projected.reinitialized);
  EXPECT_DOUBLE_EQ(0.0, projected.speed);
}

TEST(EgoFrenetProjectorTest, RejectsInvalidConfiguration)
{
  auto converter = makeSquareConverter();
  const auto infinity = std::numeric_limits<double>::infinity();

  EXPECT_THROW(
    frenet_runtime::EgoFrenetProjector(
      converter,
      frenet_runtime::EgoFrenetProjectorConfig{0.0, 0.2, 0.3}),
    std::invalid_argument);

  EXPECT_THROW(
    frenet_runtime::EgoFrenetProjector(
      converter,
      frenet_runtime::EgoFrenetProjectorConfig{0.4, 0.3, 0.2}),
    std::invalid_argument);

  EXPECT_THROW(
    frenet_runtime::EgoFrenetProjector(
      converter,
      frenet_runtime::EgoFrenetProjectorConfig{0.4, 0.3, infinity}),
    std::invalid_argument);

  EXPECT_THROW(
    frenet_runtime::EgoFrenetProjector(
      converter,
      frenet_runtime::EgoFrenetProjectorConfig{0.4, 0.3, 0.4, 0.0}),
    std::invalid_argument);
}
