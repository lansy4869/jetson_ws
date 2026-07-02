#include "frenet_mpc/local_trajectory_adapter.hpp"
#include "frenet_mpc/mpc_utils.hpp"

#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <frenet_interfaces/msg/frenet_local_trajectory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <OsqpEigen/OsqpEigen.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using namespace frenet_mpc;

}  // namespace


class FrenetMPCNode : public rclcpp::Node
{
public:
  FrenetMPCNode()
  : Node("frenet_mpc_node")
  {
    declareParameters();
    loadParameters();
    validateParameters();

    ox_.resize(config_.TK + 1);
    oy_.resize(config_.TK + 1);
    ov_.resize(config_.TK + 1);
    oyaw_.resize(config_.TK + 1);
    state_predict_ = Eigen::MatrixXd::Zero(config_.NXK, config_.TK + 1);

    mpcProbInit();
    trajectory_adapter_ = std::make_unique<LocalTrajectoryAdapter>(adapter_config_);

    parameter_callback_handle_ = add_on_set_parameters_callback(
      std::bind(&FrenetMPCNode::parametersCallback, this, std::placeholders::_1));

    odom_subscriber_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 25,
      std::bind(&FrenetMPCNode::odomCallback, this, std::placeholders::_1));

    local_trajectory_subscriber_ =
      create_subscription<frenet_interfaces::msg::FrenetLocalTrajectory>(
        local_trajectory_topic_,
        rclcpp::QoS(10).reliable().durability_volatile(),
        [this](frenet_interfaces::msg::FrenetLocalTrajectory::ConstSharedPtr msg) {
          last_local_trajectory_ = *msg;
          has_local_trajectory_ = true;
        });

    drive_publisher_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
      drive_topic_, 25);
    predicted_path_publisher_ = create_publisher<nav_msgs::msg::Path>(
      predicted_path_topic_, 10);
    ref_path_publisher_ = create_publisher<nav_msgs::msg::Path>(
      ref_path_topic_, 10);

    RCLCPP_INFO(get_logger(),
      "Frenet MPC ready: TK=%d, DTK=%.2f, local_topic=%s",
      config_.TK, config_.DTK, local_trajectory_topic_.c_str());
  }

private:
  // ========================================================================
  // Parameter handling
  // ========================================================================

  void declareParameters()
  {
    declare_parameter("odom_topic", "/pf/pose/odom");
    declare_parameter("local_trajectory_topic", "/frenet/local_trajectory");
    declare_parameter("drive_topic", "/drive");
    declare_parameter("predicted_path_topic", "/frenet_mpc/predicted_path");
    declare_parameter("ref_path_topic", "/frenet_mpc/ref_path");
    declare_parameter("global_frame", "map");
    declare_parameter("car_frame", "base_link");
    declare_parameter("horizon_TK", 12);
    declare_parameter("DTK", 0.1);
    declare_parameter("Q_x", 13.5); declare_parameter("Q_y", 13.5);
    declare_parameter("Q_v", 5.5);  declare_parameter("Q_yaw", 13.0);
    declare_parameter("Qf_x", 13.5); declare_parameter("Qf_y", 13.5);
    declare_parameter("Qf_v", 5.5);  declare_parameter("Qf_yaw", 13.0);
    declare_parameter("R_accel", 0.01); declare_parameter("R_steer", 25.0);
    declare_parameter("Rd_accel", 0.01); declare_parameter("Rd_steer", 50.0);
    declare_parameter("wheelbase", 0.42);
    declare_parameter("max_steer", 0.4120);
    declare_parameter("max_dsteer", 3.2);
    declare_parameter("max_speed", 2.5); declare_parameter("min_speed", 0.0);
    declare_parameter("max_accel", 2.0);
    declare_parameter("stale_timeout_s", 0.5);
    declare_parameter("min_sampling_speed", 0.2);
    declare_parameter("min_trajectory_points", 13);
    declare_parameter("publish_predicted_path", true);
    declare_parameter("publish_ref_path", true);
  }

  void loadParameters()
  {
    odom_topic_ = get_parameter("odom_topic").as_string();
    local_trajectory_topic_ = get_parameter("local_trajectory_topic").as_string();
    drive_topic_ = get_parameter("drive_topic").as_string();
    predicted_path_topic_ = get_parameter("predicted_path_topic").as_string();
    ref_path_topic_ = get_parameter("ref_path_topic").as_string();
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
    stale_timeout_s_ = get_parameter("stale_timeout_s").as_double();
    adapter_config_.horizon = config_.TK;
    adapter_config_.dt = config_.DTK;
    adapter_config_.min_sampling_speed = get_parameter("min_sampling_speed").as_double();
    adapter_config_.min_trajectory_points = get_parameter("min_trajectory_points").as_int();
    publish_predicted_path_ = get_parameter("publish_predicted_path").as_bool();
    publish_ref_path_ = get_parameter("publish_ref_path").as_bool();

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

    if (config_.TK < 2) throw std::invalid_argument("horizon_TK must be >= 2");
    if (config_.DTK <= 0.0) throw std::invalid_argument("DTK must be > 0");
    if (config_.WB <= 0.0) throw std::invalid_argument("wheelbase must be > 0");
    if (config_.MAX_SPEED < config_.MIN_SPEED)
      throw std::invalid_argument("max_speed must be >= min_speed");
    if (config_.MIN_SPEED < 0.0)
      throw std::invalid_argument("min_speed must be >= 0");
    if (config_.MAX_ACCEL <= 0.0)
      throw std::invalid_argument("max_accel must be > 0");
    if (config_.MAX_DSTEER <= 0.0)
      throw std::invalid_argument("max_dsteer must be > 0");
    if (config_.MAX_STEER <= 0.0)
      throw std::invalid_argument("max_steer must be > 0");
    if (odom_topic_.empty() || local_trajectory_topic_.empty() || drive_topic_.empty() ||
      predicted_path_topic_.empty() || ref_path_topic_.empty())
    {
      throw std::invalid_argument("frenet_mpc topics must not be empty");
    }
    if (!std::isfinite(stale_timeout_s_) || stale_timeout_s_ <= 0.0) {
      throw std::invalid_argument("stale_timeout_s must be positive and finite");
    }
    if (!std::isfinite(adapter_config_.min_sampling_speed) ||
      adapter_config_.min_sampling_speed < 0.0)
    {
      throw std::invalid_argument("min_sampling_speed must be non-negative and finite");
    }
    if (adapter_config_.min_trajectory_points < config_.TK + 1) {
      throw std::invalid_argument("min_trajectory_points must be at least horizon_TK + 1");
    }

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

  void publishRefPath(
    const Eigen::MatrixXd & ref_traj,
    const builtin_interfaces::msg::Time & stamp)
  {
    if (!publish_ref_path_) return;
    if (ref_traj.cols() == 0) return;

    nav_msgs::msg::Path path;
    path.header.frame_id = global_frame_;
    path.header.stamp = stamp;
    path.poses.reserve(ref_traj.cols());

    for (int i = 0; i < ref_traj.cols(); ++i) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = ref_traj(X, i);
      pose.pose.position.y = ref_traj(Y, i);
      pose.pose.position.z = 0.0;
      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, ref_traj(YAW, i));
      pose.pose.orientation = tf2::toMsg(q);
      path.poses.push_back(pose);
    }

    ref_path_publisher_->publish(path);
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

    if (!has_local_trajectory_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "No local trajectory received, publishing stop");
      publishStopCommand();
      return;
    }

    if (!isFresh(last_local_trajectory_.header.stamp, msg->header.stamp)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "Local trajectory is stale, publishing stop");
      publishStopCommand();
      return;
    }

    const auto adapter_result = trajectory_adapter_->buildReferenceTrajectory(
      last_local_trajectory_, Pose2D{pose.x, pose.y, pose.yaw});
    if (!adapter_result.ok) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "Local trajectory adapter failed: %s", adapter_result.reason.c_str());
      publishStopCommand();
      return;
    }

    Eigen::MatrixXd ref_traj = adapter_result.ref_traj;
    unwrapYawReference(ref_traj, x0(YAW));
    publishRefPath(ref_traj, msg->header.stamp);

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

  bool isFresh(
    const builtin_interfaces::msg::Time & trajectory_stamp,
    const builtin_interfaces::msg::Time & odom_stamp) const
  {
    const double trajectory_time =
      static_cast<double>(trajectory_stamp.sec) +
      static_cast<double>(trajectory_stamp.nanosec) * 1.0e-9;
    const double odom_time =
      static_cast<double>(odom_stamp.sec) +
      static_cast<double>(odom_stamp.nanosec) * 1.0e-9;
    const double age = odom_time - trajectory_time;
    return std::isfinite(age) && age >= 0.0 && age <= stale_timeout_s_;
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
  std::string odom_topic_;
  std::string local_trajectory_topic_;
  std::string drive_topic_;
  std::string predicted_path_topic_;
  std::string ref_path_topic_;
  std::string global_frame_;
  std::string car_frame_;
  double stale_timeout_s_{0.5};
  bool publish_predicted_path_{true};
  bool publish_ref_path_{true};
  AdapterConfig adapter_config_;
  std::unique_ptr<LocalTrajectoryAdapter> trajectory_adapter_;
  bool has_local_trajectory_{false};
  frenet_interfaces::msg::FrenetLocalTrajectory last_local_trajectory_;

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
  rclcpp::Subscription<frenet_interfaces::msg::FrenetLocalTrajectory>::SharedPtr
    local_trajectory_subscriber_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
    drive_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr predicted_path_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr ref_path_publisher_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
    parameter_callback_handle_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FrenetMPCNode>());
  rclcpp::shutdown();
  return 0;
}
