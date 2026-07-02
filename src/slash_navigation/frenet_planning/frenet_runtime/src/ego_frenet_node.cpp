#include "frenet_runtime/ego_frenet_projector.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <frenet_interfaces/msg/frenet_ego_state.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <track_spline/frenet_converter.hpp>

#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

std::string stripLeadingSlashes(const std::string & value)
{
  const auto first_non_slash = value.find_first_not_of('/');
  if (first_non_slash == std::string::npos) {
    return "";
  }
  return value.substr(first_non_slash);
}

bool isFinite(double value)
{
  return std::isfinite(value);
}

frenet_interfaces::msg::FrenetEgoState makeInvalidState(const std_msgs::msg::Header & header)
{
  frenet_interfaces::msg::FrenetEgoState state;
  state.header = header;
  state.valid = false;
  state.reinitialized = false;
  state.s = 0.0;
  state.s_unwrapped = 0.0;
  state.d = 0.0;
  state.yaw_error = 0.0;
  state.speed = 0.0;
  return state;
}

}  // namespace

namespace frenet_runtime
{

class EgoFrenetNode : public rclcpp::Node
{
public:
  EgoFrenetNode()
  : Node("ego_frenet_node")
  {
    declareParameters();
    loadParameters();
    projector_ = std::make_unique<EgoFrenetProjector>(loadConverter(), loadProjectorConfig());

    parameter_callback_handle_ = add_on_set_parameters_callback(
      std::bind(&EgoFrenetNode::parametersCallback, this, std::placeholders::_1));

    ego_state_publisher_ = create_publisher<frenet_interfaces::msg::FrenetEgoState>(
      ego_state_topic_,
      rclcpp::QoS(10).reliable().durability_volatile());

    odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&EgoFrenetNode::odomCallback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "ego_frenet_node ready with centerline '%s', odom='%s', ego_state='%s', global_frame='%s'",
      centerline_csv_.c_str(),
      odom_topic_.c_str(),
      ego_state_topic_.c_str(),
      global_frame_.c_str());
  }

private:
  void declareParameters()
  {
    declare_parameter("centerline_csv", "");
    declare_parameter("odom_topic", "/pf/pose/odom");
    declare_parameter("ego_state_topic", "/frenet/ego_state");
    declare_parameter("global_frame", "map");
    declare_parameter("near_search_radius_m", 3.0);
    declare_parameter("max_near_projection_distance_m", 2.0);
    declare_parameter("max_global_projection_distance_m", 3.0);
    declare_parameter("max_projection_yaw_error_rad", 1.5708);
  }

  EgoFrenetProjectorConfig loadProjectorConfig()
  {
    EgoFrenetProjectorConfig config;
    config.near_search_radius_m = near_search_radius_m_;
    config.max_near_projection_distance_m = max_near_projection_distance_m_;
    config.max_global_projection_distance_m = max_global_projection_distance_m_;
    config.max_projection_yaw_error_rad = max_projection_yaw_error_rad_;
    return config;
  }

  track_spline::FrenetConverter loadConverter()
  {
    track_spline::FrenetConverter converter;
    converter.loadCsv(centerline_csv_);
    return converter;
  }

  void loadParameters()
  {
    centerline_csv_ = get_parameter("centerline_csv").as_string();
    if (centerline_csv_.empty()) {
      centerline_csv_ = ament_index_cpp::get_package_share_directory("csv_data") + "/8flab.csv";
    }

    odom_topic_ = get_parameter("odom_topic").as_string();
    ego_state_topic_ = get_parameter("ego_state_topic").as_string();
    global_frame_ = stripLeadingSlashes(get_parameter("global_frame").as_string());
    near_search_radius_m_ = get_parameter("near_search_radius_m").as_double();
    max_near_projection_distance_m_ =
      get_parameter("max_near_projection_distance_m").as_double();
    max_global_projection_distance_m_ =
      get_parameter("max_global_projection_distance_m").as_double();
    max_projection_yaw_error_rad_ =
      get_parameter("max_projection_yaw_error_rad").as_double();

    if (centerline_csv_.empty()) {
      throw std::invalid_argument("ego_frenet_node: centerline_csv must resolve to a file");
    }
    if (odom_topic_.empty()) {
      throw std::invalid_argument("ego_frenet_node: odom_topic must not be empty");
    }
    if (ego_state_topic_.empty()) {
      throw std::invalid_argument("ego_frenet_node: ego_state_topic must not be empty");
    }
    if (global_frame_.empty()) {
      throw std::invalid_argument("ego_frenet_node: global_frame must not be empty");
    }
  }

  rcl_interfaces::msg::SetParametersResult parametersCallback(
    const std::vector<rclcpp::Parameter> & /*parameters*/)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = false;
    result.reason = "startup-only";
    return result;
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    const auto invalid_state = makeInvalidState(msg->header);

    if (stripLeadingSlashes(msg->header.frame_id) != global_frame_) {
      projector_->reset();
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        5000,
        "Rejected odometry with frame_id '%s'; expected '%s'",
        msg->header.frame_id.c_str(),
        global_frame_.c_str());
      ego_state_publisher_->publish(invalid_state);
      return;
    }

    const auto & position = msg->pose.pose.position;
    const auto & orientation_msg = msg->pose.pose.orientation;
    const double speed = msg->twist.twist.linear.x;
    if (
      !isFinite(position.x) ||
      !isFinite(position.y) ||
      !isFinite(speed) ||
      !isFinite(orientation_msg.x) ||
      !isFinite(orientation_msg.y) ||
      !isFinite(orientation_msg.z) ||
      !isFinite(orientation_msg.w))
    {
      projector_->reset();
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        5000,
        "Rejected odometry with non-finite position, velocity, or quaternion");
      ego_state_publisher_->publish(invalid_state);
      return;
    }

    tf2::Quaternion orientation;
    tf2::fromMsg(orientation_msg, orientation);
    if (orientation.length2() <= 1.0e-12) {
      projector_->reset();
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        5000,
        "Rejected odometry with near-zero quaternion norm");
      ego_state_publisher_->publish(invalid_state);
      return;
    }

    const double yaw = tf2::getYaw(orientation);
    if (!isFinite(yaw)) {
      projector_->reset();
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        5000,
        "Rejected odometry with non-finite yaw");
      ego_state_publisher_->publish(invalid_state);
      return;
    }

    const auto projected = projector_->project(position.x, position.y, yaw, speed);
    if (!projected.valid) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        5000,
        "Odometry projection to Frenet reference failed; publishing invalid ego state");
    }

    frenet_interfaces::msg::FrenetEgoState state;
    state.header = msg->header;
    state.valid = projected.valid;
    state.reinitialized = projected.reinitialized;
    if (projected.valid) {
      state.s = projected.s;
      state.s_unwrapped = projected.s_unwrapped;
      state.d = projected.d;
      state.yaw_error = projected.yaw_error;
      state.speed = projected.speed;
    }
    ego_state_publisher_->publish(state);
  }

  std::string centerline_csv_;
  std::string odom_topic_;
  std::string ego_state_topic_;
  std::string global_frame_;
  double near_search_radius_m_{0.0};
  double max_near_projection_distance_m_{0.0};
  double max_global_projection_distance_m_{0.0};
  double max_projection_yaw_error_rad_{0.0};
  std::unique_ptr<EgoFrenetProjector> projector_;
  rclcpp::Publisher<frenet_interfaces::msg::FrenetEgoState>::SharedPtr ego_state_publisher_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
};

}  // namespace frenet_runtime

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  int exit_code = 0;
  try {
    auto node = std::make_shared<frenet_runtime::EgoFrenetNode>();
    rclcpp::spin(node);
  } catch (const std::exception & exception) {
    RCLCPP_FATAL(
      rclcpp::get_logger("ego_frenet_node"),
      "Failed to start ego_frenet_node: %s",
      exception.what());
    exit_code = 1;
  } catch (...) {
    RCLCPP_FATAL(
      rclcpp::get_logger("ego_frenet_node"),
      "Failed to start ego_frenet_node: unknown exception");
    exit_code = 1;
  }

  rclcpp::shutdown();
  return exit_code;
}
