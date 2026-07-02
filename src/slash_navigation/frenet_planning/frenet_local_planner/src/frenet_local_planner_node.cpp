#include "frenet_local_planner/state_machine.hpp"
#include "frenet_local_planner/trajectory_generator.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <frenet_interfaces/msg/frenet_ego_state.hpp>
#include <frenet_interfaces/msg/frenet_local_trajectory.hpp>
#include <frenet_interfaces/msg/frenet_planner_state.hpp>
#include <frenet_interfaces/msg/reactive_advice.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <track_spline/frenet_converter.hpp>

#include <cmath>
#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

geometry_msgs::msg::Quaternion yawToQuaternion(double yaw)
{
  geometry_msgs::msg::Quaternion quaternion;
  quaternion.x = 0.0;
  quaternion.y = 0.0;
  quaternion.z = std::sin(0.5 * yaw);
  quaternion.w = std::cos(0.5 * yaw);
  return quaternion;
}

double nodeTimeSeconds(const rclcpp::Time & time)
{
  return static_cast<double>(time.nanoseconds()) * 1.0e-9;
}

}  // namespace

namespace frenet_local_planner
{

class FrenetLocalPlannerNode : public rclcpp::Node
{
public:
  FrenetLocalPlannerNode()
  : Node("frenet_local_planner_node")
  {
    declareParameters();
    loadParameters();
    validateParameters();

    converter_.loadCsv(centerline_csv_);
    state_machine_ = std::make_unique<StateMachine>(planner_config_);
    trajectory_generator_ = std::make_unique<TrajectoryGenerator>(trajectory_config_);

    ego_subscriber_ = create_subscription<frenet_interfaces::msg::FrenetEgoState>(
      ego_state_topic_,
      rclcpp::QoS(10).reliable().durability_volatile(),
      [this](frenet_interfaces::msg::FrenetEgoState::ConstSharedPtr msg) {
        last_ego_ = *msg;
        has_ego_ = true;
      });

    advice_subscriber_ = create_subscription<frenet_interfaces::msg::ReactiveAdvice>(
      reactive_advice_topic_,
      rclcpp::SensorDataQoS(),
      [this](frenet_interfaces::msg::ReactiveAdvice::ConstSharedPtr msg) {
        last_advice_ = *msg;
        has_advice_ = true;
      });

    planner_state_publisher_ =
      create_publisher<frenet_interfaces::msg::FrenetPlannerState>(
        planner_state_topic_, 10);
    local_trajectory_publisher_ =
      create_publisher<frenet_interfaces::msg::FrenetLocalTrajectory>(
        local_trajectory_topic_, 10);
    local_path_publisher_ = create_publisher<nav_msgs::msg::Path>(local_path_topic_, 10);

    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / planner_rate_hz_),
      std::bind(&FrenetLocalPlannerNode::onTimer, this));

    RCLCPP_INFO(
      get_logger(),
      "frenet_local_planner_node ready with centerline '%s'",
      centerline_csv_.c_str());
  }

private:
  void declareParameters()
  {
    declare_parameter("centerline_csv", "");
    declare_parameter("ego_state_topic", "/frenet/ego_state");
    declare_parameter("reactive_advice_topic", "/frenet/reactive_advice");
    declare_parameter("planner_state_topic", "/frenet/planner_state");
    declare_parameter("local_trajectory_topic", "/frenet/local_trajectory");
    declare_parameter("local_path_topic", "/frenet/local_path");
    declare_parameter("global_frame", "map");
    declare_parameter("planner_rate_hz", 20.0);
    declare_parameter("trajectory_points", 20);
    declare_parameter("trajectory_ds", 0.25);
    declare_parameter("overtake_offset_m", 0.45);
    declare_parameter("center_tolerance_m", 0.10);
    declare_parameter("lane_change_length_m", 2.0);
    declare_parameter("stale_timeout_s", 0.5);
    declare_parameter("normal_speed", 2.0);
    declare_parameter("follow_speed", 0.8);
    declare_parameter("overtake_speed", 1.5);
    declare_parameter("emergency_speed", 0.0);
    declare_parameter("overtake_confidence_min", 0.6);
    declare_parameter("emergency_risk_threshold", 0.9);
    declare_parameter("emergency_front_clearance_m", 0.30);
  }

  void loadParameters()
  {
    centerline_csv_ = get_parameter("centerline_csv").as_string();
    if (centerline_csv_.empty()) {
      centerline_csv_ =
        ament_index_cpp::get_package_share_directory("csv_data") + "/8flab.csv";
    }

    ego_state_topic_ = get_parameter("ego_state_topic").as_string();
    reactive_advice_topic_ = get_parameter("reactive_advice_topic").as_string();
    planner_state_topic_ = get_parameter("planner_state_topic").as_string();
    local_trajectory_topic_ = get_parameter("local_trajectory_topic").as_string();
    local_path_topic_ = get_parameter("local_path_topic").as_string();
    global_frame_ = get_parameter("global_frame").as_string();
    planner_rate_hz_ = get_parameter("planner_rate_hz").as_double();

    trajectory_config_.trajectory_points = get_parameter("trajectory_points").as_int();
    trajectory_config_.trajectory_ds = get_parameter("trajectory_ds").as_double();
    trajectory_config_.lane_change_length_m =
      get_parameter("lane_change_length_m").as_double();

    planner_config_.overtake_offset_m = get_parameter("overtake_offset_m").as_double();
    planner_config_.center_tolerance_m = get_parameter("center_tolerance_m").as_double();
    planner_config_.stale_timeout_s = get_parameter("stale_timeout_s").as_double();
    planner_config_.normal_speed = get_parameter("normal_speed").as_double();
    planner_config_.follow_speed = get_parameter("follow_speed").as_double();
    planner_config_.overtake_speed = get_parameter("overtake_speed").as_double();
    planner_config_.emergency_speed = get_parameter("emergency_speed").as_double();
    planner_config_.overtake_confidence_min =
      get_parameter("overtake_confidence_min").as_double();
    planner_config_.emergency_risk_threshold =
      get_parameter("emergency_risk_threshold").as_double();
    planner_config_.emergency_front_clearance_m =
      get_parameter("emergency_front_clearance_m").as_double();
  }

  void validateParameters() const
  {
    if (centerline_csv_.empty()) {
      throw std::invalid_argument("centerline_csv must resolve to a file");
    }
    if (
      ego_state_topic_.empty() ||
      reactive_advice_topic_.empty() ||
      planner_state_topic_.empty() ||
      local_trajectory_topic_.empty() ||
      local_path_topic_.empty())
    {
      throw std::invalid_argument("planner topics must not be empty");
    }
    if (!std::isfinite(planner_rate_hz_) || planner_rate_hz_ <= 0.0) {
      throw std::invalid_argument("planner_rate_hz must be positive and finite");
    }
    if (trajectory_config_.trajectory_points < 2) {
      throw std::invalid_argument("trajectory_points must be at least 2");
    }
    if (!std::isfinite(trajectory_config_.trajectory_ds) ||
      trajectory_config_.trajectory_ds <= 0.0)
    {
      throw std::invalid_argument("trajectory_ds must be positive and finite");
    }
    if (!std::isfinite(trajectory_config_.lane_change_length_m) ||
      trajectory_config_.lane_change_length_m <= 0.0)
    {
      throw std::invalid_argument("lane_change_length_m must be positive and finite");
    }
    if (!std::isfinite(planner_config_.overtake_offset_m) ||
      planner_config_.overtake_offset_m <= 0.0)
    {
      throw std::invalid_argument("overtake_offset_m must be positive and finite");
    }
    if (!std::isfinite(planner_config_.center_tolerance_m) ||
      planner_config_.center_tolerance_m < 0.0)
    {
      throw std::invalid_argument("center_tolerance_m must be non-negative and finite");
    }
    if (!std::isfinite(planner_config_.stale_timeout_s) ||
      planner_config_.stale_timeout_s <= 0.0)
    {
      throw std::invalid_argument("stale_timeout_s must be positive and finite");
    }
    if (!std::isfinite(planner_config_.normal_speed) ||
      !std::isfinite(planner_config_.follow_speed) ||
      !std::isfinite(planner_config_.overtake_speed) ||
      !std::isfinite(planner_config_.emergency_speed) ||
      planner_config_.normal_speed < 0.0 ||
      planner_config_.follow_speed < 0.0 ||
      planner_config_.overtake_speed < 0.0 ||
      planner_config_.emergency_speed < 0.0)
    {
      throw std::invalid_argument("planner speeds must be non-negative and finite");
    }
    if (!std::isfinite(planner_config_.overtake_confidence_min) ||
      planner_config_.overtake_confidence_min < 0.0 ||
      planner_config_.overtake_confidence_min > 1.0)
    {
      throw std::invalid_argument("overtake_confidence_min must be in [0, 1]");
    }
    if (!std::isfinite(planner_config_.emergency_risk_threshold) ||
      planner_config_.emergency_risk_threshold < 0.0 ||
      planner_config_.emergency_risk_threshold > 1.0)
    {
      throw std::invalid_argument("emergency_risk_threshold must be in [0, 1]");
    }
    if (!std::isfinite(planner_config_.emergency_front_clearance_m) ||
      planner_config_.emergency_front_clearance_m < 0.0)
    {
      throw std::invalid_argument("emergency_front_clearance_m must be non-negative and finite");
    }
  }

  void onTimer()
  {
    frenet_interfaces::msg::FrenetEgoState ego = last_ego_;
    frenet_interfaces::msg::ReactiveAdvice advice = last_advice_;
    if (!has_ego_) {
      ego.valid = false;
    }
    if (!has_advice_) {
      advice.valid = false;
    }

    auto decision = state_machine_->decide(ego, advice, nodeTimeSeconds(now()));
    frenet_interfaces::msg::FrenetLocalTrajectory trajectory;
    try {
      trajectory = buildLocalTrajectory(ego, decision);
      if (decision.valid && !trajectory.valid) {
        decision = emergencyDecision();
        trajectory = buildLocalTrajectory(ego, decision);
      }
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Failed to generate Frenet local trajectory: %s", error.what());
      decision = emergencyDecision();
      trajectory = buildLocalTrajectory(ego, decision);
    }

    publishPlannerState(decision);
    local_trajectory_publisher_->publish(trajectory);
    local_path_publisher_->publish(toPath(trajectory));
  }

  void publishPlannerState(const PlannerDecision & decision)
  {
    frenet_interfaces::msg::FrenetPlannerState state;
    state.header.frame_id = global_frame_;
    state.header.stamp = now();
    state.valid = decision.valid;
    state.state = decision.state;
    state.target_d = decision.target_d;
    state.target_speed = decision.target_speed;
    planner_state_publisher_->publish(state);
  }

  PlannerDecision emergencyDecision() const
  {
    PlannerDecision decision;
    decision.valid = false;
    decision.state = frenet_interfaces::msg::FrenetPlannerState::EMERGENCY;
    decision.target_d = 0.0;
    decision.target_speed = planner_config_.emergency_speed;
    return decision;
  }

  frenet_interfaces::msg::FrenetLocalTrajectory buildLocalTrajectory(
    const frenet_interfaces::msg::FrenetEgoState & ego,
    const PlannerDecision & decision)
  {
    frenet_interfaces::msg::FrenetLocalTrajectory trajectory;
    trajectory.header = ego.header;
    if (trajectory.header.frame_id.empty()) {
      trajectory.header.frame_id = global_frame_;
      trajectory.header.stamp = now();
    }
    trajectory.valid = decision.valid;
    trajectory.state = decision.state;
    trajectory.target_d = decision.target_d;
    trajectory.target_speed = decision.target_speed;
    if (decision.valid) {
      trajectory.points = trajectory_generator_->generate(ego, decision, converter_);
      trajectory.valid = !trajectory.points.empty();
    }

    return trajectory;
  }

  nav_msgs::msg::Path toPath(
    const frenet_interfaces::msg::FrenetLocalTrajectory & trajectory) const
  {
    nav_msgs::msg::Path path;
    path.header = trajectory.header;
    path.poses.reserve(trajectory.points.size());
    for (const auto & point : trajectory.points) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = point.x;
      pose.pose.position.y = point.y;
      pose.pose.position.z = 0.0;
      pose.pose.orientation = yawToQuaternion(point.yaw);
      path.poses.push_back(pose);
    }
    return path;
  }

  std::string centerline_csv_;
  std::string ego_state_topic_;
  std::string reactive_advice_topic_;
  std::string planner_state_topic_;
  std::string local_trajectory_topic_;
  std::string local_path_topic_;
  std::string global_frame_;
  double planner_rate_hz_{20.0};
  PlannerConfig planner_config_;
  TrajectoryConfig trajectory_config_;
  track_spline::FrenetConverter converter_;
  std::unique_ptr<StateMachine> state_machine_;
  std::unique_ptr<TrajectoryGenerator> trajectory_generator_;
  bool has_ego_{false};
  bool has_advice_{false};
  frenet_interfaces::msg::FrenetEgoState last_ego_;
  frenet_interfaces::msg::ReactiveAdvice last_advice_;
  rclcpp::Subscription<frenet_interfaces::msg::FrenetEgoState>::SharedPtr ego_subscriber_;
  rclcpp::Subscription<frenet_interfaces::msg::ReactiveAdvice>::SharedPtr advice_subscriber_;
  rclcpp::Publisher<frenet_interfaces::msg::FrenetPlannerState>::SharedPtr
    planner_state_publisher_;
  rclcpp::Publisher<frenet_interfaces::msg::FrenetLocalTrajectory>::SharedPtr
    local_trajectory_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_path_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace frenet_local_planner

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<frenet_local_planner::FrenetLocalPlannerNode>());
  rclcpp::shutdown();
  return 0;
}
