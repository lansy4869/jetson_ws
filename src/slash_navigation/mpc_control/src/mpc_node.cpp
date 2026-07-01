#include "mpc_control/mpc_utils.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/msg/marker.hpp>

#include <OsqpEigen/OsqpEigen.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "track_spline/closed_loop_trajectory.hpp"

namespace
{

using namespace mpc_control;

constexpr double kYawTolerance = 1.0e-6;

struct Waypoint
{
  double x;
  double y;
  double yaw;
  double curvature;
  double arc_length;
};

struct Pose2D
{
  double x;
  double y;
  double yaw;
};

// ---- CSV parsing helpers (same pattern as pure_pursuit) ----

std::string trim(const std::string & value)
{
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return "";
  const auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

std::string toLower(const std::string & value)
{
  std::string lowered;
  lowered.reserve(value.size());
  std::transform(value.begin(), value.end(), std::back_inserter(lowered),
    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return lowered;
}

bool isXYYawHeader(const std::vector<std::string> & columns)
{
  return columns.size() == 3 &&
    toLower(columns[0]) == "x" &&
    toLower(columns[1]) == "y" &&
    toLower(columns[2]) == "yaw";
}

std::vector<std::string> splitCsvLine(const std::string & line)
{
  std::vector<std::string> columns;
  std::stringstream stream(line);
  std::string column;
  while (std::getline(stream, column, ',')) {
    columns.push_back(trim(column));
  }
  if (!line.empty() && line.back() == ',') {
    columns.push_back("");
  }
  return columns;
}

double parseFiniteDouble(const std::string & text, std::size_t line_number,
                         const char * field_name)
{
  try {
    std::size_t parsed_length = 0;
    const double value = std::stod(text, &parsed_length);
    if (parsed_length != text.size() || !std::isfinite(value)) {
      throw std::invalid_argument("not finite");
    }
    return value;
  } catch (const std::exception &) {
    std::ostringstream msg;
    msg << "Invalid " << field_name << " on CSV line " << line_number
        << ": '" << text << "'";
    throw std::runtime_error(msg.str());
  }
}

double distance2d(double x1, double y1, double x2, double y2)
{
  return std::hypot(x2 - x1, y2 - y1);
}

}  // namespace


class MPCNode : public rclcpp::Node
{
public:
  MPCNode()
  : Node("mpc_node")
  {
    declareParameters();
    loadParameters();
    validateParameters();
    loadWaypoints();

    // Size solution vectors
    ox_.resize(config_.TK + 1);
    oy_.resize(config_.TK + 1);
    ov_.resize(config_.TK + 1);
    oyaw_.resize(config_.TK + 1);
    state_predict_ = Eigen::MatrixXd::Zero(config_.NXK, config_.TK + 1);

    // Build CiMPCC speed profile from waypoint curvatures
    buildSpeedProfile();

    // Build track_spline trajectory for spline mode
    buildSplineTrajectory();

    // Initialize MPC QP problem
    mpcProbInit();

    // Set up parameter callback
    parameter_callback_handle_ = add_on_set_parameters_callback(
      std::bind(&MPCNode::parametersCallback, this, std::placeholders::_1));

    // ROS interfaces
    odom_subscriber_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 25,
      std::bind(&MPCNode::odomCallback, this, std::placeholders::_1));

    drive_publisher_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
      drive_topic_, 25);

    predicted_path_publisher_ = create_publisher<nav_msgs::msg::Path>(
      "/mpc/predicted_path", 10);

    ref_trajectory_publisher_ = create_publisher<nav_msgs::msg::Path>(
      "/mpc/ref_trajectory", rclcpp::QoS(1).transient_local());

    publishRefTrajectoryPath();

    RCLCPP_INFO(get_logger(),
      "MPC ready: %zu waypoints, mode=%s, TK=%d, DTK=%.2f",
      waypoints_.size(), trajectory_mode_.c_str(), config_.TK, config_.DTK);
  }

private:
  // ========================================================================
  // Parameter handling
  // ========================================================================

  void declareParameters()
  {
    declare_parameter("waypoint_csv", "");
    declare_parameter("trajectory_mode", "discrete");
    declare_parameter("spline_sample_resolution", 0.05);
    declare_parameter("odom_topic", "/pf/pose/odom");
    declare_parameter("drive_topic", "/drive");
    declare_parameter("global_frame", "map");
    declare_parameter("car_frame", "base_link");
    declare_parameter("horizon_TK", 8);
    declare_parameter("DTK", 0.1);
    declare_parameter("Q_x", 13.5); declare_parameter("Q_y", 13.5);
    declare_parameter("Q_v", 5.5);  declare_parameter("Q_yaw", 13.0);
    declare_parameter("Qf_x", 13.5); declare_parameter("Qf_y", 13.5);
    declare_parameter("Qf_v", 5.5);  declare_parameter("Qf_yaw", 13.0);
    declare_parameter("R_accel", 0.01); declare_parameter("R_steer", 100.0);
    declare_parameter("Rd_accel", 0.01); declare_parameter("Rd_steer", 100.0);
    declare_parameter("wheelbase", 0.33);
    declare_parameter("max_steer", 0.4189);
    declare_parameter("max_dsteer", 3.14);
    declare_parameter("max_speed", 2.5); declare_parameter("min_speed", 0.5);
    declare_parameter("max_accel", 3.0);
    declare_parameter("curvature_smooth_window", 3);
    declare_parameter("cimpcc_alpha", 1.5);
    declare_parameter("publish_predicted_path", true);
    declare_parameter("publish_ref_trajectory", true);
  }

  void loadParameters()
  {
    waypoint_csv_ = get_parameter("waypoint_csv").as_string();
    if (waypoint_csv_.empty()) {
      waypoint_csv_ = ament_index_cpp::get_package_share_directory("csv_data")
                      + "/8flab.csv";
    }
    trajectory_mode_ = get_parameter("trajectory_mode").as_string();
    config_.spline_sample_resolution =
      get_parameter("spline_sample_resolution").as_double();
    odom_topic_ = get_parameter("odom_topic").as_string();
    drive_topic_ = get_parameter("drive_topic").as_string();
    global_frame_ = get_parameter("global_frame").as_string();
    car_frame_ = get_parameter("car_frame").as_string();
    config_.TK = get_parameter("horizon_TK").as_int();
    config_.DTK = get_parameter("DTK").as_double();
    config_.WB = get_parameter("wheelbase").as_double();
    config_.MAX_STEER = get_parameter("max_steer").as_double();
    config_.MIN_STEER = -config_.MAX_STEER;
    config_.MAX_DSTEER = get_parameter("max_dsteer").as_double();
    config_.MAX_SPEED = get_parameter("max_speed").as_double();
    config_.MIN_SPEED = get_parameter("min_speed").as_double();
    config_.MAX_ACCEL = get_parameter("max_accel").as_double();
    config_.curvature_smooth_window =
      get_parameter("curvature_smooth_window").as_int();
    config_.cimpcc_alpha = get_parameter("cimpcc_alpha").as_double();
    publish_predicted_path_ = get_parameter("publish_predicted_path").as_bool();
    publish_ref_trajectory_ = get_parameter("publish_ref_trajectory").as_bool();

    // Build cost matrices from diagonal parameters
    {
      Eigen::Matrix4d Q;
      Q.setZero();
      Q(X, X) = get_parameter("Q_x").as_double();
      Q(Y, Y) = get_parameter("Q_y").as_double();
      Q(V, V) = get_parameter("Q_v").as_double();
      Q(YAW, YAW) = get_parameter("Q_yaw").as_double();
      config_.Qk = Q;
    }
    {
      Eigen::Matrix4d Qf;
      Qf.setZero();
      Qf(X, X) = get_parameter("Qf_x").as_double();
      Qf(Y, Y) = get_parameter("Qf_y").as_double();
      Qf(V, V) = get_parameter("Qf_v").as_double();
      Qf(YAW, YAW) = get_parameter("Qf_yaw").as_double();
      config_.Qfk = Qf;
    }
    {
      Eigen::Matrix2d R;
      R.setZero();
      R(ACCEL, ACCEL) = get_parameter("R_accel").as_double();
      R(STEER, STEER) = get_parameter("R_steer").as_double();
      config_.Rk = R;
    }
    {
      Eigen::Matrix2d Rd;
      Rd.setZero();
      Rd(ACCEL, ACCEL) = get_parameter("Rd_accel").as_double();
      Rd(STEER, STEER) = get_parameter("Rd_steer").as_double();
      config_.Rdk = Rd;
    }
  }

  void validateParameters() const
  {
    auto checkFinite = [](const std::string & name, double value) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument(name + " must be finite");
      }
    };
    checkFinite("DTK", config_.DTK);
    checkFinite("wheelbase", config_.WB);
    checkFinite("max_steer", config_.MAX_STEER);
    checkFinite("max_dsteer", config_.MAX_DSTEER);
    checkFinite("max_speed", config_.MAX_SPEED);
    checkFinite("min_speed", config_.MIN_SPEED);
    checkFinite("max_accel", config_.MAX_ACCEL);
    checkFinite("cimpcc_alpha", config_.cimpcc_alpha);
    checkFinite("spline_sample_resolution", config_.spline_sample_resolution);

    if (trajectory_mode_ != "discrete" && trajectory_mode_ != "spline") {
      throw std::invalid_argument("trajectory_mode must be discrete or spline");
    }
    if (config_.TK < 2) throw std::invalid_argument("horizon_TK must be >= 2");
    if (config_.DTK <= 0.0) throw std::invalid_argument("DTK must be > 0");
    if (config_.WB <= 0.0) throw std::invalid_argument("wheelbase must be > 0");
    if (config_.MAX_SPEED < config_.MIN_SPEED)
      throw std::invalid_argument("max_speed must be >= min_speed");
    if (config_.MIN_SPEED < 0.0)
      throw std::invalid_argument("min_speed must be >= 0");
    if (config_.cimpcc_alpha <= 0.0)
      throw std::invalid_argument("cimpcc_alpha must be > 0");
    if (config_.MAX_ACCEL <= 0.0)
      throw std::invalid_argument("max_accel must be > 0");
    if (config_.MAX_DSTEER <= 0.0)
      throw std::invalid_argument("max_dsteer must be > 0");
    if (config_.MAX_STEER <= 0.0)
      throw std::invalid_argument("max_steer must be > 0");
    if (config_.spline_sample_resolution <= 0.0)
      throw std::invalid_argument("spline_sample_resolution must be > 0");

    // Cost matrix diagonals: any negative entry makes the Hessian indefinite,
    // turning the QP into an unbounded maximization problem.
    auto checkDiag = [](const std::string & name, double value) {
      if (!std::isfinite(value))
        throw std::invalid_argument(name + " must be finite");
      if (value < 0.0)
        throw std::invalid_argument(name + " must be >= 0 (negative cost makes QP non-convex)");
    };
    checkDiag("Q_x", config_.Qk(X, X));
    checkDiag("Q_y", config_.Qk(Y, Y));
    checkDiag("Q_v", config_.Qk(V, V));
    checkDiag("Q_yaw", config_.Qk(YAW, YAW));
    checkDiag("Qf_x", config_.Qfk(X, X));
    checkDiag("Qf_y", config_.Qfk(Y, Y));
    checkDiag("Qf_v", config_.Qfk(V, V));
    checkDiag("Qf_yaw", config_.Qfk(YAW, YAW));
    checkDiag("R_accel", config_.Rk(ACCEL, ACCEL));
    checkDiag("R_steer", config_.Rk(STEER, STEER));
    checkDiag("Rd_accel", config_.Rdk(ACCEL, ACCEL));
    checkDiag("Rd_steer", config_.Rdk(STEER, STEER));

    // At least one control must be penalised to keep the QP bounded.
    if (config_.Rk(ACCEL, ACCEL) == 0.0 && config_.Rk(STEER, STEER) == 0.0 &&
        config_.Rdk(ACCEL, ACCEL) == 0.0 && config_.Rdk(STEER, STEER) == 0.0) {
      throw std::invalid_argument("at least one of R_* or Rd_* must be > 0");
    }
  }

  rcl_interfaces::msg::SetParametersResult parametersCallback(
    const std::vector<rclcpp::Parameter> & /*parameters*/)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = false;
    result.reason = "MPC parameters can only be changed at launch";
    return result;
  }

  // ========================================================================
  // Waypoint loading (same pattern as pure_pursuit)
  // ========================================================================

  void loadWaypoints()
  {
    std::ifstream csv_file(waypoint_csv_);
    if (!csv_file.is_open()) {
      throw std::runtime_error("Unable to open waypoint CSV: " + waypoint_csv_);
    }

    waypoints_.clear();
    std::string line;
    std::size_t line_number = 0;
    bool is_first_non_empty_line = true;

    while (std::getline(csv_file, line)) {
      ++line_number;
      const std::string trimmed = trim(line);
      if (trimmed.empty()) continue;

      const auto columns = splitCsvLine(trimmed);
      if (columns.size() != 3) {
        std::ostringstream msg;
        msg << "CSV line " << line_number << " must contain exactly 3 columns";
        throw std::runtime_error(msg.str());
      }

      try {
        Waypoint wp{parseFiniteDouble(columns[0], line_number, "x"),
                     parseFiniteDouble(columns[1], line_number, "y"),
                     parseFiniteDouble(columns[2], line_number, "yaw"),
                     0.0, 0.0};

        if (wp.yaw < -kPi - kYawTolerance || wp.yaw > kPi + kYawTolerance) {
          std::ostringstream msg;
          msg << "Yaw outside [-pi, pi] on line " << line_number
              << ": " << wp.yaw;
          throw std::runtime_error(msg.str());
        }
        wp.yaw = clampValue(wp.yaw, -kPi, kPi);
        waypoints_.push_back(wp);
        is_first_non_empty_line = false;
      } catch (const std::runtime_error &) {
        if (is_first_non_empty_line && isXYYawHeader(columns)) {
          is_first_non_empty_line = false;
          continue;
        }
        throw;
      }
    }

    if (waypoints_.size() < 3) {
      throw std::runtime_error("Waypoint CSV must contain at least 3 waypoints");
    }

    computeDerivedWaypointFields();
  }

  void computeDerivedWaypointFields()
  {
    waypoints_.front().arc_length = 0.0;
    for (std::size_t i = 1; i < waypoints_.size(); ++i) {
      const auto & prev = waypoints_[i - 1];
      waypoints_[i].arc_length = prev.arc_length +
        distance2d(prev.x, prev.y, waypoints_[i].x, waypoints_[i].y);
    }

    for (std::size_t i = 0; i < waypoints_.size(); ++i) {
      const std::size_t next_idx = (i + 1) % waypoints_.size();
      const double seg_dist = distance2d(
        waypoints_[i].x, waypoints_[i].y,
        waypoints_[next_idx].x, waypoints_[next_idx].y);
      if (seg_dist <= kNearZero) {
        waypoints_[i].curvature = 0.0;
        continue;
      }
      waypoints_[i].curvature = wrapAngle(
        waypoints_[next_idx].yaw - waypoints_[i].yaw) / seg_dist;
    }

    const double total_length = waypoints_.back().arc_length +
      distance2d(waypoints_.back().x, waypoints_.back().y,
                 waypoints_.front().x, waypoints_.front().y);

    RCLCPP_INFO(get_logger(),
      "Loaded %zu waypoints, total_length=%.2f m",
      waypoints_.size(), total_length);
  }

  // ========================================================================
  // Speed profile & spline trajectory builders
  // ========================================================================

  void buildSpeedProfile()
  {
    std::vector<double> curvatures;
    curvatures.reserve(waypoints_.size());
    for (const auto & wp : waypoints_) {
      curvatures.push_back(wp.curvature);
    }

    speed_profile_ = CiMPCCSpeedProfile(
      curvatures, config_.MIN_SPEED, config_.MAX_SPEED,
      config_.cimpcc_alpha, config_.curvature_smooth_window);

    RCLCPP_INFO(get_logger(),
      "CiMPCC speed profile built: size=%zu, v_min=%.2f, v_max=%.2f, alpha=%.1f",
      speed_profile_.size(), config_.MIN_SPEED, config_.MAX_SPEED,
      config_.cimpcc_alpha);
  }

  void buildSplineTrajectory()
  {
    std::vector<track_spline::TrackPoint> points;
    points.reserve(waypoints_.size());
    for (const auto & wp : waypoints_) {
      points.push_back({wp.x, wp.y, wp.yaw});
    }
    spline_trajectory_.build(points);

    // Build spline speed profile (curvature sampled at dl intervals)
    std::vector<double> spline_curvatures;
    const double total_length = spline_trajectory_.length();
    const double dl = config_.spline_sample_resolution;
    const int n_samples = std::max(3, static_cast<int>(total_length / dl));

    for (int i = 0; i < n_samples; ++i) {
      const double s = total_length * static_cast<double>(i) /
                       static_cast<double>(n_samples);
      try {
        const auto sample = spline_trajectory_.evaluate(s);
        spline_curvatures.push_back(sample.curvature);
      } catch (...) {
        spline_curvatures.push_back(0.0);
      }
    }

    spline_speed_profile_ = CiMPCCSpeedProfile(
      spline_curvatures, config_.MIN_SPEED, config_.MAX_SPEED,
      config_.cimpcc_alpha, config_.curvature_smooth_window);

    RCLCPP_INFO(get_logger(),
      "Spline trajectory built: length=%.2f m, samples=%d",
      total_length, n_samples);
  }

  // ========================================================================
  // Reference trajectory builders
  // ========================================================================

  Eigen::MatrixXd buildDiscreteRefTrajectory(const Pose2D & pose)
  {
    const std::size_t n = waypoints_.size();
    Eigen::VectorXd cx(n), cy(n), cv(n), cyaw(n);
    for (std::size_t i = 0; i < n; ++i) {
      cx(i) = waypoints_[i].x;
      cy(i) = waypoints_[i].y;
      cv(i) = speed_profile_[i];
      cyaw(i) = waypoints_[i].yaw;
    }
    return calcInterpolatedRefTrajectory(
      pose.x, pose.y, cx, cy, cv, cyaw, config_.DTK, config_.TK);
  }

  Eigen::MatrixXd buildSplineRefTrajectory(const Pose2D & pose)
  {
    Eigen::MatrixXd ref_traj = Eigen::MatrixXd::Zero(NX, config_.TK + 1);

    if (spline_trajectory_.empty()) return ref_traj;

    // Project onto spline to find starting arc-length
    auto proj = spline_trajectory_.project(pose.x, pose.y);
    double s_current = proj.s;
    const double total_len = spline_trajectory_.length();
    const int n_speed_samples = static_cast<int>(spline_speed_profile_.size());

    // Helper: get reference speed at arc-length s by linear interpolation
    auto getSpeedAtS = [&](double s) -> double {
      double wrapped = std::fmod(s, total_len);
      if (wrapped < 0.0) wrapped += total_len;
      if (wrapped >= total_len) wrapped = 0.0;
      double frac = wrapped / total_len;
      double idx_f = frac * (n_speed_samples - 1);
      int idx = static_cast<int>(idx_f);
      int idx_next = std::min(idx + 1, n_speed_samples - 1);
      double t = idx_f - idx;
      return (1.0 - t) * spline_speed_profile_[idx] +
             t * spline_speed_profile_[idx_next];
    };

    // Step forward T steps.
    // Each column t stores (x, y, v, yaw) at the SAME arc-length s_current,
    // so the MPC state cost penalises a consistent reference point in time.
    for (int t = 0; t <= config_.TK; ++t) {
      if (t == 0) {
        ref_traj(X, 0) = proj.sample.x;
        ref_traj(Y, 0) = proj.sample.y;
        ref_traj(V, 0) = getSpeedAtS(s_current);
        ref_traj(YAW, 0) = proj.sample.yaw;
      } else {
        // Advance arc-length using speed at current position
        s_current += getSpeedAtS(s_current) * config_.DTK;

        auto sample = spline_trajectory_.evaluate(s_current);
        ref_traj(X, t) = sample.x;
        ref_traj(Y, t) = sample.y;
        ref_traj(V, t) = getSpeedAtS(s_current);
        ref_traj(YAW, t) = sample.yaw;
      }
    }

    return ref_traj;
  }

  void publishRefTrajectoryPath()
  {
    if (!publish_ref_trajectory_) return;

    nav_msgs::msg::Path path;
    path.header.frame_id = global_frame_;
    path.header.stamp = now();
    path.poses.reserve(waypoints_.size());

    for (const auto & wp : waypoints_) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = wp.x;
      pose.pose.position.y = wp.y;
      pose.pose.position.z = 0.0;
      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, wp.yaw);
      pose.pose.orientation = tf2::toMsg(q);
      path.poses.push_back(pose);
    }

    ref_trajectory_publisher_->publish(path);
  }

  void publishPredictedPath(const Eigen::MatrixXd & predicted,
                            const rclcpp::Time & stamp)
  {
    if (!publish_predicted_path_) return;
    if (predicted.cols() == 0) return;

    nav_msgs::msg::Path path;
    path.header.frame_id = global_frame_;
    path.header.stamp = stamp;
    path.poses.reserve(predicted.cols());

    for (int i = 0; i < predicted.cols(); ++i) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = predicted(X, i);
      pose.pose.position.y = predicted(Y, i);
      pose.pose.position.z = 0.0;
      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, predicted(YAW, i));
      pose.pose.orientation = tf2::toMsg(q);
      path.poses.push_back(pose);
    }

    predicted_path_publisher_->publish(path);
  }

  // ========================================================================
  // MPC QP Methods
  // ========================================================================

  void mpcProbInit()
  {
    const int n_vars = numVars(config_.TK);

    // Initialize warm-start trajectories
    xk_ = Eigen::MatrixXd::Zero(config_.NXK, config_.TK + 1);
    uk_ = Eigen::MatrixXd::Zero(config_.NU_, config_.TK);
    ref_traj_k_ = Eigen::MatrixXd::Zero(config_.NXK, config_.TK + 1);
    x0k_ = Eigen::Vector4d::Zero();

    // Linearized dynamics matrices (one per timestep)
    A_block_.resize(config_.TK);
    B_block_.resize(config_.TK);
    C_block_.resize(config_.TK);
    for (int t = 0; t < config_.TK; ++t) {
      A_block_[t] = Eigen::Matrix4d::Identity();
      B_block_[t] = Eigen::Matrix<double, NX, NU>::Zero();
      C_block_[t] = Eigen::Vector4d::Zero();
    }

    // ---- Build Hessian P (constant, built once) ----
    std::vector<Eigen::Triplet<double>> P_triplets;

    // Control cost: Rk on each u_t
    for (int t = 0; t < config_.TK; ++t) {
      for (int r = 0; r < NU; ++r) {
        for (int c = 0; c < NU; ++c) {
          if (std::abs(config_.Rk(r, c)) > kNearZero) {
            P_triplets.emplace_back(inputIdx(t, r, config_.TK),
                                    inputIdx(t, c, config_.TK),
                                    config_.Rk(r, c));
          }
        }
      }
    }

    // Control rate cost: Rdk on (u_{t+1} - u_t)
    for (int t = 0; t < config_.TK - 1; ++t) {
      for (int r = 0; r < NU; ++r) {
        for (int c = 0; c < NU; ++c) {
          if (std::abs(config_.Rdk(r, c)) > kNearZero) {
            double val = config_.Rdk(r, c);
            int i1 = inputIdx(t, r, config_.TK);
            int i2 = inputIdx(t + 1, r, config_.TK);
            int j1 = inputIdx(t, c, config_.TK);
            int j2 = inputIdx(t + 1, c, config_.TK);
            P_triplets.emplace_back(i1, j1, val);    // u0^T Rd u0
            P_triplets.emplace_back(i2, j2, val);    // u1^T Rd u1
            P_triplets.emplace_back(i1, j2, -val);   // -u0^T Rd u1
            P_triplets.emplace_back(i2, j1, -val);   // -u1^T Rd u0
          }
        }
      }
    }

    // State cost: Qk on each (x_t - x_ref_t), Qfk on terminal
    for (int t = 0; t < config_.TK; ++t) {
      const Eigen::Matrix4d & Q = config_.Qk;
      for (int r = 0; r < NX; ++r) {
        for (int c = 0; c < NX; ++c) {
          if (std::abs(Q(r, c)) > kNearZero) {
            P_triplets.emplace_back(stateIdx(t, r, config_.TK),
                                    stateIdx(t, c, config_.TK), Q(r, c));
          }
        }
      }
    }
    {
      const Eigen::Matrix4d & Qf = config_.Qfk;
      for (int r = 0; r < NX; ++r) {
        for (int c = 0; c < NX; ++c) {
          if (std::abs(Qf(r, c)) > kNearZero) {
            P_triplets.emplace_back(
              stateIdx(config_.TK, r, config_.TK),
              stateIdx(config_.TK, c, config_.TK), Qf(r, c));
          }
        }
      }
    }

    P_.resize(n_vars, n_vars);
    P_.setFromTriplets(P_triplets.begin(), P_triplets.end());

    // ---- Count constraints ----
    int n_dynamics = config_.NXK * config_.TK;
    int n_init = config_.NXK;
    int n_speed_bounds = (config_.TK + 1) * 1;
    int n_accel_bounds = config_.TK * 1;
    int n_steer_bounds = config_.TK * 1;
    int n_steer_rate = (config_.TK - 1) * 1;

    int n_constraints = n_dynamics + n_init + n_speed_bounds +
      n_accel_bounds + n_steer_bounds + n_steer_rate;

    // ---- Build constraint matrix A structure ----
    std::vector<Eigen::Triplet<double>> A_triplets;
    int row = 0;

    // Dynamics: x_{t+1} - A_t*x_t - B_t*u_t = C_t
    for (int t = 0; t < config_.TK; ++t) {
      // x_{t+1} (identity)
      for (int r = 0; r < NX; ++r) {
        A_triplets.emplace_back(row + r, stateIdx(t + 1, r, config_.TK), 1.0);
      }
      // -A_t (full block, values updated each solve)
      for (int r = 0; r < NX; ++r) {
        for (int c = 0; c < NX; ++c) {
          A_triplets.emplace_back(row + r, stateIdx(t, c, config_.TK), 0.0);
        }
      }
      // -B_t (full block, values updated each solve)
      for (int r = 0; r < NX; ++r) {
        for (int c = 0; c < NU; ++c) {
          A_triplets.emplace_back(row + r, inputIdx(t, c, config_.TK), 0.0);
        }
      }
      row += NX;
    }

    // Initial state: x_0 = x0k_
    for (int r = 0; r < NX; ++r) {
      A_triplets.emplace_back(row + r, stateIdx(0, r, config_.TK), 1.0);
    }
    row += NX;

    // Speed bounds: v_min <= v_t <= v_max
    for (int t = 0; t <= config_.TK; ++t) {
      A_triplets.emplace_back(row, stateIdx(t, V, config_.TK), 1.0);
      ++row;
    }

    // Accel bounds: -a_max <= accel_t <= a_max
    for (int t = 0; t < config_.TK; ++t) {
      A_triplets.emplace_back(row, inputIdx(t, ACCEL, config_.TK), 1.0);
      ++row;
    }

    // Steer bounds: -δ_max <= steer_t <= δ_max
    for (int t = 0; t < config_.TK; ++t) {
      A_triplets.emplace_back(row, inputIdx(t, STEER, config_.TK), 1.0);
      ++row;
    }

    // Steer rate: |steer_{t+1} - steer_t| <= dδ_max * DTK
    for (int t = 0; t < config_.TK - 1; ++t) {
      A_triplets.emplace_back(row, inputIdx(t + 1, STEER, config_.TK), 1.0);
      A_triplets.emplace_back(row, inputIdx(t, STEER, config_.TK), -1.0);
      ++row;
    }

    A_.resize(n_constraints, n_vars);
    A_.setFromTriplets(A_triplets.begin(), A_triplets.end());

    // Initialize bound vectors
    l_ = Eigen::VectorXd::Zero(n_constraints);
    u_ = Eigen::VectorXd::Zero(n_constraints);

    // ---- Initialize OSQP solver ----
    solver_.settings()->setVerbosity(false);
    solver_.settings()->setWarmStart(true);
    solver_.settings()->setPolish(true);

    solver_.data()->setNumberOfVariables(n_vars);
    solver_.data()->setNumberOfConstraints(n_constraints);
    solver_.data()->setHessianMatrix(P_);

    Eigen::VectorXd initial_grad = Eigen::VectorXd::Zero(n_vars);
    solver_.data()->setGradient(initial_grad);
    solver_.data()->setLinearConstraintsMatrix(A_);
    solver_.data()->setLowerBound(l_);
    solver_.data()->setUpperBound(u_);

    if (!solver_.initSolver()) {
      throw std::runtime_error("OSQP solver failed to initialize");
    }

    RCLCPP_INFO(get_logger(),
      "MPC QP initialized: %d vars, %d constraints",
      n_vars, n_constraints);
  }

  bool updateConstraintMatrix()
  {
    std::vector<Eigen::Triplet<double>> A_triplets;

    for (int k = 0; k < A_.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(A_, k); it; ++it) {
        double val = it.value();
        int r = static_cast<int>(it.row());
        int c = static_cast<int>(it.col());

        // Check if this entry is in the dynamics section (rows 0 .. NX*TK - 1)
        if (r < config_.NXK * config_.TK) {
          int t = r / NX;
          int local_row = r % NX;

          // x_{t+1} entry: identity block
          if (c >= stateIdx(t + 1, 0, config_.TK) &&
              c <  stateIdx(t + 2, 0, config_.TK)) {
            val = 1.0;
          }
          // x_t entry: -A_t block
          else if (c >= stateIdx(t, 0, config_.TK) &&
                   c <  stateIdx(t + 1, 0, config_.TK)) {
            int comp = c - stateIdx(t, 0, config_.TK);
            val = -A_block_[t](local_row, comp);
          }
          // u_t entry: -B_t block
          else if (c >= inputIdx(t, 0, config_.TK) &&
                   c <  inputIdx(t + 1, 0, config_.TK)) {
            int comp = c - inputIdx(t, 0, config_.TK);
            val = -B_block_[t](local_row, comp);
          }
        }

        // Always emit the triplet to preserve sparsity pattern
        A_triplets.emplace_back(r, c, val);
      }
    }

    A_.setFromTriplets(A_triplets.begin(), A_triplets.end());
    return solver_.updateLinearConstraintsMatrix(A_);
  }

  bool updateBounds(const Eigen::Vector4d & x0)
  {
    int row = 0;

    // Dynamics: C_t vectors (equality, l = u = C_t component)
    for (int t = 0; t < config_.TK; ++t) {
      for (int r = 0; r < NX; ++r) {
        l_(row) = C_block_[t](r);
        u_(row) = C_block_[t](r);
        ++row;
      }
    }

    // Initial state: fixed to measurement
    for (int r = 0; r < NX; ++r) {
      l_(row) = x0(r);
      u_(row) = x0(r);
      ++row;
    }

    // Speed bounds
    for (int t = 0; t <= config_.TK; ++t) {
      l_(row) = reachableSpeedLowerBound(x0(V), t, config_);
      u_(row) = reachableSpeedUpperBound(x0(V), t, config_);
      ++row;
    }

    // Accel bounds
    for (int t = 0; t < config_.TK; ++t) {
      l_(row) = -config_.MAX_ACCEL;
      u_(row) = config_.MAX_ACCEL;
      ++row;
    }

    // Steer bounds
    for (int t = 0; t < config_.TK; ++t) {
      l_(row) = config_.MIN_STEER;
      u_(row) = config_.MAX_STEER;
      ++row;
    }

    // Steer rate bounds
    double dsteer_bound = config_.MAX_DSTEER * config_.DTK;
    for (int t = 0; t < config_.TK - 1; ++t) {
      l_(row) = -dsteer_bound;
      u_(row) = dsteer_bound;
      ++row;
    }

    return solver_.updateBounds(l_, u_);
  }

  bool updateGradient(const Eigen::MatrixXd & ref_traj)
  {
    const int n_vars = numVars(config_.TK);
    Eigen::VectorXd q = Eigen::VectorXd::Zero(n_vars);

    // State tracking: -Qk * x_ref_t
    for (int t = 0; t < config_.TK; ++t) {
      Eigen::Vector4d x_ref = ref_traj.col(t);
      Eigen::Vector4d Qx_ref = -(config_.Qk * x_ref);
      for (int r = 0; r < NX; ++r) {
        q(stateIdx(t, r, config_.TK)) = Qx_ref(r);
      }
    }

    // Terminal state: -Qfk * x_ref_T
    {
      Eigen::Vector4d x_ref_T = ref_traj.col(config_.TK);
      Eigen::Vector4d Qfx_ref = -(config_.Qfk * x_ref_T);
      for (int r = 0; r < NX; ++r) {
        q(stateIdx(config_.TK, r, config_.TK)) = Qfx_ref(r);
      }
    }

    return solver_.updateGradient(q);
  }

  Eigen::MatrixXd predictMotion(const Eigen::Vector4d & x0,
                                const std::vector<double> & oa,
                                const std::vector<double> & od)
  {
    Eigen::MatrixXd path_predict = Eigen::MatrixXd::Zero(config_.NXK,
                                                         config_.TK + 1);
    path_predict.col(0) = x0;

    State state{x0(X), x0(Y), x0(V), x0(YAW)};

    for (int t = 1; t <= config_.TK; ++t) {
      state = updateState(state, oa[t - 1], od[t - 1], config_);
      path_predict(X, t) = state.x;
      path_predict(Y, t) = state.y;
      path_predict(V, t) = state.v;
      path_predict(YAW, t) = state.yaw;
    }

    return path_predict;
  }

  bool mpcProbSolve(const Eigen::MatrixXd & ref_traj,
                    const Eigen::MatrixXd & path_predict,
                    const Eigen::Vector4d & x0)
  {
    x0k_ = x0;

    // Update linearized dynamics at each timestep along predicted path
    // Linearize around the warm-start steering trajectory (odelta_),
    // clamped to valid range for consistent Jacobian evaluation.
    for (int t = 0; t < config_.TK; ++t) {
      double delta_op = clampValue(odelta_[t], config_.MIN_STEER,
                                   config_.MAX_STEER);
      getLinearizedModel(path_predict(V, t), path_predict(YAW, t), delta_op,
                         config_, A_block_[t], B_block_[t], C_block_[t]);
    }

    // Update constraint matrix with new A_t, B_t
    if (!updateConstraintMatrix()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "updateConstraintMatrix failed");
      return false;
    }

    // Update bounds with new C_t and x0
    if (!updateBounds(x0)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "updateBounds failed");
      return false;
    }

    // Update gradient with new reference trajectory
    if (!updateGradient(ref_traj)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "updateGradient failed");
      return false;
    }

    // Solve
    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
      return false;
    }
    const auto status = solver_.getStatus();
    if (!isOsqpSolvedStatus(status)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "OSQP did not solve MPC QP successfully, status=%d",
        static_cast<int>(status));
      return false;
    }

    return true;
  }

  bool linearMpcControl(const Eigen::MatrixXd & ref_traj,
                        const Eigen::Vector4d & x0)
  {
    if (odelta_.size() != static_cast<std::size_t>(config_.TK)) {
      odelta_.assign(config_.TK, 0.0);
      oa_.assign(config_.TK, 0.0);
    }

    // Predict motion for linearization points
    Eigen::MatrixXd path_predict = predictMotion(x0, oa_, odelta_);

    // Solve MPC
    if (!mpcProbSolve(ref_traj, path_predict, x0)) {
      return false;
    }

    // Extract solution
    Eigen::VectorXd solution = solver_.getSolution();

    for (int t = 0; t <= config_.TK; ++t) {
      ox_(t) = solution(stateIdx(t, X, config_.TK));
      oy_(t) = solution(stateIdx(t, Y, config_.TK));
      ov_(t) = solution(stateIdx(t, V, config_.TK));
      oyaw_(t) = solution(stateIdx(t, YAW, config_.TK));
    }

    for (int t = 0; t < config_.TK; ++t) {
      oa_[t] = solution(inputIdx(t, ACCEL, config_.TK));
      odelta_[t] = solution(inputIdx(t, STEER, config_.TK));
    }

    // Reject solution with non-finite entries: a numerically unstable QP
    // can return NaN/Inf even when the solver reports NoError.
    for (int t = 0; t < config_.TK; ++t) {
      if (!std::isfinite(oa_[t]) || !std::isfinite(odelta_[t])) {
        return false;
      }
    }

    // Build state_predict_ from solution
    state_predict_ = Eigen::MatrixXd::Zero(config_.NXK, config_.TK + 1);
    for (int t = 0; t <= config_.TK; ++t) {
      state_predict_(X, t) = ox_(t);
      state_predict_(Y, t) = oy_(t);
      state_predict_(V, t) = ov_(t);
      state_predict_(YAW, t) = oyaw_(t);
    }

    return true;
  }

  // ========================================================================
  // Main callback
  // ========================================================================

  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    // Validate odom
    if (!validateOdom(*msg)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "Invalid odom, publishing stop");
      publishStopCommand();
      return;
    }

    // Extract pose and state
    const Pose2D pose{msg->pose.pose.position.x,
                       msg->pose.pose.position.y,
                       tf2::getYaw(msg->pose.pose.orientation)};

    const double current_speed = std::hypot(
      msg->twist.twist.linear.x, msg->twist.twist.linear.y);

    Eigen::Vector4d x0;
    x0 << pose.x, pose.y, current_speed, pose.yaw;

    // Build reference trajectory
    Eigen::MatrixXd ref_traj;
    if (trajectory_mode_ == "spline") {
      ref_traj = buildSplineRefTrajectory(pose);
    } else {
      ref_traj = buildDiscreteRefTrajectory(pose);
    }
    unwrapYawReference(ref_traj, x0(YAW));

    // Solve MPC
    if (!linearMpcControl(ref_traj, x0)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "MPC solve failed, publishing stop");
      publishStopCommand();
      return;
    }

    // Publish drive command — both steer and speed are clamped and
    // finite-checked as defense-in-depth against solver numerical issues.
    double steer_cmd = clampValue(odelta_[0], config_.MIN_STEER,
                                  config_.MAX_STEER);
    double speed_cmd = current_speed + oa_[0] * config_.DTK;
    speed_cmd = clampValue(
      speed_cmd, commandSpeedLowerBound(current_speed, config_), config_.MAX_SPEED);

    if (!std::isfinite(steer_cmd) || !std::isfinite(speed_cmd)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "Non-finite control command detected, publishing stop");
      publishStopCommand();
      return;
    }

    ackermann_msgs::msg::AckermannDriveStamped drive_msg;
    drive_msg.header.stamp = now();
    drive_msg.header.frame_id = car_frame_;
    drive_msg.drive.steering_angle = steer_cmd;
    drive_msg.drive.speed = speed_cmd;
    drive_publisher_->publish(drive_msg);

    // Publish predicted path for visualization
    publishPredictedPath(state_predict_, msg->header.stamp);
  }

  bool validateOdom(const nav_msgs::msg::Odometry & msg) const
  {
    if (msg.header.frame_id != global_frame_) return false;
    if (msg.child_frame_id != car_frame_) return false;

    const auto & pos = msg.pose.pose.position;
    const auto & orient = msg.pose.pose.orientation;
    const auto & linear = msg.twist.twist.linear;
    const auto & angular = msg.twist.twist.angular;

    const double vals[] = {
      pos.x, pos.y, pos.z,
      orient.x, orient.y, orient.z, orient.w,
      linear.x, linear.y, linear.z,
      angular.x, angular.y, angular.z
    };

    for (double v : vals) {
      if (!std::isfinite(v)) return false;
    }

    double qnorm = std::sqrt(orient.x * orient.x + orient.y * orient.y +
                             orient.z * orient.z + orient.w * orient.w);
    if (qnorm <= kNearZero) return false;

    return true;
  }

  void publishStopCommand()
  {
    ackermann_msgs::msg::AckermannDriveStamped drive_msg;
    drive_msg.header.stamp = now();
    drive_msg.header.frame_id = car_frame_;
    drive_msg.drive.steering_angle = 0.0;
    drive_msg.drive.speed = 0.0;
    drive_publisher_->publish(drive_msg);
  }

  // ========================================================================
  // Member variables
  // ========================================================================

  // Configuration
  Config config_;
  std::string waypoint_csv_;
  std::string trajectory_mode_;
  std::string odom_topic_;
  std::string drive_topic_;
  std::string global_frame_;
  std::string car_frame_;
  bool publish_predicted_path_;
  bool publish_ref_trajectory_;

  // Trajectory data
  std::vector<Waypoint> waypoints_;
  std::vector<double> speed_profile_;          // CiMPCC-mapped speeds (discrete)
  std::vector<double> spline_speed_profile_;   // CiMPCC-mapped speeds (spline)
  track_spline::ClosedLoopTrajectory spline_trajectory_;

  // MPC warm-start buffers
  std::vector<double> oa_;       // acceleration trajectory [TK]
  std::vector<double> odelta_;   // steering trajectory [TK]

  // MPC solution storage (sized in constructor to TK+1)
  Eigen::VectorXd ox_;
  Eigen::VectorXd oy_;
  Eigen::VectorXd ov_;
  Eigen::VectorXd oyaw_;
  Eigen::MatrixXd state_predict_;  // (NX, TK+1) solved trajectory

  // QP matrices
  Eigen::SparseMatrix<double> P_;   // Hessian (constant)
  Eigen::SparseMatrix<double> A_;   // Constraint matrix
  Eigen::VectorXd l_, u_;           // Constraint bounds

  // Linearized dynamics (updated each solve)
  std::vector<Eigen::Matrix4d> A_block_;
  std::vector<Eigen::Matrix<double, NX, NU>> B_block_;
  std::vector<Eigen::Vector4d> C_block_;

  // OSQP workspace variables
  Eigen::MatrixXd xk_;          // state trajectory warm-start
  Eigen::MatrixXd uk_;          // input trajectory warm-start
  Eigen::MatrixXd ref_traj_k_;  // reference trajectory
  Eigen::Vector4d x0k_;         // initial state

  OsqpEigen::Solver solver_;

  // ROS interfaces
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
    drive_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr predicted_path_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr ref_trajectory_publisher_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
    parameter_callback_handle_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MPCNode>());
  rclcpp::shutdown();
  return 0;
}
