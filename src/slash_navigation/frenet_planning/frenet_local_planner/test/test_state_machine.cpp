#include "frenet_local_planner/state_machine.hpp"

#include <frenet_interfaces/msg/frenet_planner_state.hpp>
#include <gtest/gtest.h>

#include <limits>

namespace
{

using frenet_interfaces::msg::FrenetEgoState;
using frenet_interfaces::msg::FrenetPlannerState;
using frenet_interfaces::msg::ReactiveAdvice;

frenet_local_planner::PlannerConfig makeConfig()
{
  frenet_local_planner::PlannerConfig config;
  config.overtake_offset_m = 0.45;
  config.center_tolerance_m = 0.10;
  config.stale_timeout_s = 0.5;
  config.normal_speed = 2.0;
  config.follow_speed = 0.8;
  config.overtake_speed = 1.5;
  config.emergency_speed = 0.0;
  config.overtake_confidence_min = 0.6;
  config.emergency_risk_threshold = 0.9;
  config.emergency_front_clearance_m = 0.30;
  return config;
}

FrenetEgoState makeEgo(double stamp_seconds = 10.0)
{
  FrenetEgoState ego;
  ego.header.stamp.sec = static_cast<int32_t>(stamp_seconds);
  ego.header.stamp.nanosec = static_cast<uint32_t>(
    (stamp_seconds - static_cast<double>(ego.header.stamp.sec)) * 1.0e9);
  ego.valid = true;
  ego.reinitialized = false;
  ego.s = 1.0;
  ego.s_unwrapped = 1.0;
  ego.d = 0.0;
  ego.yaw_error = 0.0;
  ego.speed = 1.0;
  return ego;
}

ReactiveAdvice makeAdvice(double stamp_seconds = 10.0)
{
  ReactiveAdvice advice;
  advice.header.stamp.sec = static_cast<int32_t>(stamp_seconds);
  advice.header.stamp.nanosec = static_cast<uint32_t>(
    (stamp_seconds - static_cast<double>(advice.header.stamp.sec)) * 1.0e9);
  advice.valid = true;
  advice.front_blocked = false;
  advice.left_gap_available = true;
  advice.right_gap_available = true;
  advice.overtake_left_available = true;
  advice.overtake_right_available = true;
  advice.front_clearance_m = 4.0;
  advice.left_clearance_m = 4.0;
  advice.right_clearance_m = 4.0;
  advice.best_gap_angle_rad = 0.0;
  advice.best_gap_width_rad = 1.0;
  advice.best_gap_confidence = 1.0;
  advice.dynamic_obstacle_confidence = 0.0;
  advice.obstacle_risk = 0.0;
  advice.obstacle_count = 0;
  return advice;
}

}  // namespace

TEST(FrenetStateMachineTest, OpenScanTracksGlobalCenterline)
{
  frenet_local_planner::StateMachine machine(makeConfig());

  const auto decision = machine.decide(makeEgo(), makeAdvice(), 10.1);

  EXPECT_TRUE(decision.valid);
  EXPECT_EQ(FrenetPlannerState::GB_TRACK, decision.state);
  EXPECT_DOUBLE_EQ(0.0, decision.target_d);
  EXPECT_DOUBLE_EQ(2.0, decision.target_speed);
}

TEST(FrenetStateMachineTest, FrontBlockedWithoutGapFollows)
{
  auto advice = makeAdvice();
  advice.front_blocked = true;
  advice.overtake_left_available = false;
  advice.overtake_right_available = false;
  advice.best_gap_confidence = 0.1;
  advice.obstacle_risk = 0.5;
  frenet_local_planner::StateMachine machine(makeConfig());

  const auto decision = machine.decide(makeEgo(), advice, 10.1);

  EXPECT_TRUE(decision.valid);
  EXPECT_EQ(FrenetPlannerState::FOLLOW, decision.state);
  EXPECT_DOUBLE_EQ(0.0, decision.target_d);
  EXPECT_DOUBLE_EQ(0.0, decision.target_speed);
}

TEST(FrenetStateMachineTest, FrontBlockedWithLeftGapOvertakesLeft)
{
  auto advice = makeAdvice();
  advice.front_blocked = true;
  advice.overtake_left_available = true;
  advice.overtake_right_available = false;
  advice.best_gap_angle_rad = 0.4;
  advice.best_gap_confidence = 0.8;
  advice.obstacle_risk = 0.5;
  frenet_local_planner::StateMachine machine(makeConfig());

  const auto decision = machine.decide(makeEgo(), advice, 10.1);

  EXPECT_TRUE(decision.valid);
  EXPECT_EQ(FrenetPlannerState::OVERTAKE_LEFT, decision.state);
  EXPECT_DOUBLE_EQ(0.45, decision.target_d);
  EXPECT_DOUBLE_EQ(1.5, decision.target_speed);
}

TEST(FrenetStateMachineTest, FrontBlockedWithBothGapsUsesGapAngleDeadband)
{
  auto advice = makeAdvice();
  advice.front_blocked = true;
  advice.overtake_left_available = true;
  advice.overtake_right_available = true;
  advice.best_gap_confidence = 0.8;
  advice.obstacle_risk = 0.5;
  frenet_local_planner::StateMachine machine(makeConfig());

  advice.best_gap_angle_rad = -1.0e-12;
  auto decision = machine.decide(makeEgo(), advice, 10.1);
  EXPECT_TRUE(decision.valid);
  EXPECT_EQ(FrenetPlannerState::OVERTAKE_LEFT, decision.state);
  EXPECT_DOUBLE_EQ(0.45, decision.target_d);

  advice.best_gap_angle_rad = -0.2;
  decision = machine.decide(makeEgo(), advice, 10.1);
  EXPECT_TRUE(decision.valid);
  EXPECT_EQ(FrenetPlannerState::OVERTAKE_RIGHT, decision.state);
  EXPECT_DOUBLE_EQ(-0.45, decision.target_d);
}

TEST(FrenetStateMachineTest, StaleOrInvalidInputEntersEmergency)
{
  auto ego = makeEgo(9.0);
  auto advice = makeAdvice(10.0);
  frenet_local_planner::StateMachine machine(makeConfig());

  auto decision = machine.decide(ego, advice, 10.1);
  EXPECT_EQ(FrenetPlannerState::EMERGENCY, decision.state);
  EXPECT_FALSE(decision.valid);
  EXPECT_DOUBLE_EQ(0.0, decision.target_speed);

  ego = makeEgo(10.0);
  ego.valid = false;
  decision = machine.decide(ego, makeAdvice(), 10.1);
  EXPECT_EQ(FrenetPlannerState::EMERGENCY, decision.state);
  EXPECT_FALSE(decision.valid);

  ego = makeEgo(10.0);
  ego.reinitialized = true;
  decision = machine.decide(ego, makeAdvice(), 10.1);
  EXPECT_EQ(FrenetPlannerState::EMERGENCY, decision.state);
  EXPECT_FALSE(decision.valid);
}

TEST(FrenetStateMachineTest, NonFiniteEgoFieldsEnterEmergency)
{
  frenet_local_planner::StateMachine machine(makeConfig());
  auto ego = makeEgo();
  ego.s = std::numeric_limits<double>::quiet_NaN();

  auto decision = machine.decide(ego, makeAdvice(), 10.1);
  EXPECT_EQ(FrenetPlannerState::EMERGENCY, decision.state);
  EXPECT_FALSE(decision.valid);

  ego = makeEgo();
  ego.yaw_error = std::numeric_limits<double>::quiet_NaN();
  decision = machine.decide(ego, makeAdvice(), 10.1);
  EXPECT_EQ(FrenetPlannerState::EMERGENCY, decision.state);
  EXPECT_FALSE(decision.valid);

  ego = makeEgo();
  ego.speed = std::numeric_limits<double>::quiet_NaN();
  decision = machine.decide(ego, makeAdvice(), 10.1);
  EXPECT_EQ(FrenetPlannerState::EMERGENCY, decision.state);
  EXPECT_FALSE(decision.valid);
}

TEST(FrenetStateMachineTest, HighRiskOrNonFiniteAdviceEntersEmergency)
{
  auto advice = makeAdvice();
  advice.front_blocked = false;
  advice.obstacle_risk = 0.95;
  frenet_local_planner::StateMachine machine(makeConfig());

  auto decision = machine.decide(makeEgo(), advice, 10.1);
  EXPECT_EQ(FrenetPlannerState::EMERGENCY, decision.state);
  EXPECT_FALSE(decision.valid);

  advice = makeAdvice();
  advice.obstacle_risk = std::numeric_limits<double>::quiet_NaN();
  decision = machine.decide(makeEgo(), advice, 10.1);
  EXPECT_EQ(FrenetPlannerState::EMERGENCY, decision.state);
  EXPECT_FALSE(decision.valid);

  advice = makeAdvice();
  advice.best_gap_angle_rad = std::numeric_limits<double>::quiet_NaN();
  advice.front_blocked = true;
  decision = machine.decide(makeEgo(), advice, 10.1);
  EXPECT_EQ(FrenetPlannerState::EMERGENCY, decision.state);
  EXPECT_FALSE(decision.valid);
}

TEST(FrenetStateMachineTest, OutOfRangeAdviceScoresEnterEmergency)
{
  frenet_local_planner::StateMachine machine(makeConfig());

  auto advice = makeAdvice();
  advice.best_gap_confidence = 1.2;
  auto decision = machine.decide(makeEgo(), advice, 10.1);
  EXPECT_EQ(FrenetPlannerState::EMERGENCY, decision.state);
  EXPECT_FALSE(decision.valid);

  advice = makeAdvice();
  advice.dynamic_obstacle_confidence = -0.1;
  decision = machine.decide(makeEgo(), advice, 10.1);
  EXPECT_EQ(FrenetPlannerState::EMERGENCY, decision.state);
  EXPECT_FALSE(decision.valid);

  advice = makeAdvice();
  advice.front_clearance_m = -0.1;
  decision = machine.decide(makeEgo(), advice, 10.1);
  EXPECT_EQ(FrenetPlannerState::EMERGENCY, decision.state);
  EXPECT_FALSE(decision.valid);
}
