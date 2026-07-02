#ifndef FRENET_MPC_MPC_UTILS_HPP_
#define FRENET_MPC_MPC_UTILS_HPP_

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <OsqpEigen/Constants.hpp>

namespace frenet_mpc
{

// ============================================================================
// Constants
// ============================================================================
constexpr int NX = 4;  // State size: [x, y, v, yaw]
constexpr int NU = 2;  // Input size: [accel, steer_angle]
constexpr double kPi = 3.14159265358979323846;
constexpr double kNearZero = 1.0e-9;

// State indices
enum StateIdx { X = 0, Y = 1, V = 2, YAW = 3 };

// Input indices
enum InputIdx { ACCEL = 0, STEER = 1 };

// ============================================================================
// Configuration struct — values populated from ROS parameters
// ============================================================================
struct Config
{
  int NXK = NX;
  int NU_ = NU;
  int TK = 8;
  double DTK = 0.1;

  // Cost matrices (built after parameter loading)
  Eigen::Matrix4d Qk = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d Qfk = Eigen::Matrix4d::Identity();
  Eigen::Matrix2d Rk = Eigen::Matrix2d::Identity();
  Eigen::Matrix2d Rdk = Eigen::Matrix2d::Identity();

  // Vehicle
  double WB = 0.33;
  double MIN_STEER = -0.4189;
  double MAX_STEER = 0.4189;
  double MAX_DSTEER = 3.14;
  double MAX_SPEED = 2.5;
  double MIN_SPEED = 0.5;
  double MAX_ACCEL = 3.0;
};

// ============================================================================
// State representation
// ============================================================================
struct State
{
  double x = 0.0;
  double y = 0.0;
  double v = 0.0;
  double yaw = 0.0;
};

// ============================================================================
// Utility functions
// ============================================================================
inline double wrapAngle(double angle)
{
  while (angle > kPi) angle -= 2.0 * kPi;
  while (angle < -kPi) angle += 2.0 * kPi;
  return angle;
}

inline double unwrapAngleNear(double angle, double reference)
{
  return reference + wrapAngle(angle - reference);
}

inline void unwrapYawReference(Eigen::MatrixXd & ref_traj, double current_yaw)
{
  if (ref_traj.rows() <= YAW || ref_traj.cols() == 0) {
    return;
  }

  ref_traj(YAW, 0) = unwrapAngleNear(ref_traj(YAW, 0), current_yaw);
  for (int i = 1; i < ref_traj.cols(); ++i) {
    ref_traj(YAW, i) = unwrapAngleNear(ref_traj(YAW, i), ref_traj(YAW, i - 1));
  }
}

inline double clampValue(double value, double lo, double hi)
{
  if (!std::isfinite(value)) {
    return (lo <= 0.0 && hi >= 0.0) ? 0.0 : lo;
  }
  return std::max(lo, std::min(hi, value));
}

inline double nonnegativeFiniteSpeed(double speed)
{
  if (!std::isfinite(speed)) {
    return 0.0;
  }
  return std::max(0.0, speed);
}

inline double reachableSpeedLowerBound(double current_speed, int step,
                                       const Config & config)
{
  const double start_speed = nonnegativeFiniteSpeed(current_speed);
  const double dt = std::isfinite(config.DTK) ? std::max(0.0, config.DTK) : 0.0;
  const double max_accel =
    std::isfinite(config.MAX_ACCEL) ? std::max(0.0, config.MAX_ACCEL) : 0.0;
  const double min_speed =
    std::isfinite(config.MIN_SPEED) ? std::max(0.0, config.MIN_SPEED) : 0.0;
  const double reachable_upper =
    start_speed + max_accel * dt * static_cast<double>(std::max(0, step));

  return std::min(min_speed, reachable_upper);
}

inline double reachableSpeedUpperBound(double current_speed, int step,
                                       const Config & config)
{
  const double start_speed = nonnegativeFiniteSpeed(current_speed);
  const double dt = std::isfinite(config.DTK) ? std::max(0.0, config.DTK) : 0.0;
  const double max_accel =
    std::isfinite(config.MAX_ACCEL) ? std::max(0.0, config.MAX_ACCEL) : 0.0;
  const double max_speed =
    std::isfinite(config.MAX_SPEED) ? std::max(0.0, config.MAX_SPEED) : 0.0;
  const double reachable_lower =
    start_speed - max_accel * dt * static_cast<double>(std::max(0, step));

  return std::max(max_speed, reachable_lower);
}

inline double commandSpeedLowerBound(double current_speed, const Config & config)
{
  const double start_speed = nonnegativeFiniteSpeed(current_speed);
  const double min_speed =
    std::isfinite(config.MIN_SPEED) ? std::max(0.0, config.MIN_SPEED) : 0.0;
  if (min_speed <= kNearZero) {
    return 0.0;
  }
  return start_speed + kNearZero >= min_speed ? min_speed : 0.0;
}

inline bool isOsqpSolvedStatus(OsqpEigen::Status status)
{
  return status == OsqpEigen::Status::Solved ||
         status == OsqpEigen::Status::SolvedInaccurate;
}

// ============================================================================
// Vehicle Dynamics — nonlinear model for motion prediction
// ============================================================================
inline State updateState(const State & state, double accel, double steer,
                         const Config & config)
{
  State next = state;
  double delta = clampValue(steer, config.MIN_STEER, config.MAX_STEER);

  next.x = state.x + state.v * std::cos(state.yaw) * config.DTK;
  next.y = state.y + state.v * std::sin(state.yaw) * config.DTK;
  next.yaw = state.yaw + (state.v / config.WB) * std::tan(delta) * config.DTK;
  next.v = clampValue(state.v + accel * config.DTK,
                      config.MIN_SPEED, config.MAX_SPEED);

  return next;
}

// ============================================================================
// Vehicle Dynamics — linearized discrete-time model
// x_{t+1} = A * x_t + B * u_t + C
// ============================================================================
inline void getLinearizedModel(double v, double phi, double delta,
                               const Config & config,
                               Eigen::Matrix4d & A,
                               Eigen::Matrix<double, NX, NU> & B,
                               Eigen::Vector4d & C)
{
  A.setIdentity();
  A(X, V) = config.DTK * std::cos(phi);
  A(X, YAW) = -config.DTK * v * std::sin(phi);
  A(Y, V) = config.DTK * std::sin(phi);
  A(Y, YAW) = config.DTK * v * std::cos(phi);
  A(YAW, V) = config.DTK * std::tan(delta) / config.WB;

  // Guard cos²(delta) against near-zero: cos(delta) ≈ 0 only when
  // delta ≈ π/2 (≈ 1.57 rad), far outside the clamped steering range
  // (±0.42 rad). The floor is defense-in-depth against future misuse.
  double cos_delta = std::cos(delta);
  double cos_delta_sq = cos_delta * cos_delta;
  if (cos_delta_sq < kNearZero) { cos_delta_sq = kNearZero; }

  B.setZero();
  B(V, ACCEL) = config.DTK;
  B(YAW, STEER) = config.DTK * v / (config.WB * cos_delta_sq);

  C.setZero();
  C(X) = config.DTK * v * std::sin(phi) * phi;
  C(Y) = -config.DTK * v * std::cos(phi) * phi;
  C(YAW) = -config.DTK * v * delta / (config.WB * cos_delta_sq);
}

// ============================================================================
// Indexing helpers — decision variable layout:
//   z = [x_0, x_1, ..., x_T, u_0, u_1, ..., u_{T-1}]
// ============================================================================
inline int stateIdx(int t, int component, int horizon)
{
  (void)horizon;
  return t * NX + component;
}

inline int inputIdx(int t, int component, int horizon)
{
  return (horizon + 1) * NX + t * NU + component;
}

inline int numVars(int horizon)
{
  return (horizon + 1) * NX + horizon * NU;
}

}  // namespace frenet_mpc

#endif  // FRENET_MPC_MPC_UTILS_HPP_
