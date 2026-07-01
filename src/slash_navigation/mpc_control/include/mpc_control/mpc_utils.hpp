#ifndef MPC_CONTROL_MPC_UTILS_HPP_
#define MPC_CONTROL_MPC_UTILS_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <OsqpEigen/Constants.hpp>

namespace mpc_control
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

  // CiMPCC
  double cimpcc_alpha = 1.5;
  int curvature_smooth_window = 3;
  double spline_sample_resolution = 0.05;
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

// ============================================================================
// CiMPCC curvature-to-velocity mapping
// ============================================================================

/**
 * Apply CiMPCC velocity mapping to a curvature sequence.
 *
 * Algorithm (CiMPCC §III.B power-law mapping):
 *   1. |κ|_norm(i) = |κ(i)| / max(|κ|), clamped to [0, 1]
 *   2. v_raw(i) = v_max * (1 - |κ|_norm(i))^alpha
 *   3. v_ref(i) = max(v_raw(i), v_min) — floor at minimum speed
 *   4. Rolling-window average smoothing
 *   5. Clamp to [v_min, v_max]
 *
 * Power-law shape (alpha > 1): speed stays high on moderate curves,
 * drops aggressively on tight curves → more aggressive racing line
 * than the linear mapping used by vanilla curvature-speed heuristics.
 *
 * @param curvatures  Absolute curvature at each sample point
 * @param v_min       Minimum reference speed (floor)
 * @param v_max       Maximum reference speed (straight-line)
 * @param alpha       Power-law exponent, typically 1.5–2.0
 * @param smooth_window  Rolling average window (odd number, min 1)
 * @return mapped speeds, same size as input
 */
inline std::vector<double> CiMPCCSpeedProfile(
    const std::vector<double> & curvatures,
    double v_min, double v_max, double alpha, int smooth_window)
{
  const std::size_t n = curvatures.size();
  std::vector<double> speeds(n, v_min);

  if (n == 0) return speeds;

  // Step 1-2: Find max curvature and normalize
  double max_kappa = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    max_kappa = std::max(max_kappa, std::abs(curvatures[i]));
  }

  if (max_kappa <= kNearZero) {
    // Straight track — all max speed
    std::fill(speeds.begin(), speeds.end(), v_max);
    return speeds;
  }

  // Step 3: Power-law mapping with v_min floor
  for (std::size_t i = 0; i < n; ++i) {
    double kappa_norm = std::abs(curvatures[i]) / max_kappa;
    kappa_norm = clampValue(kappa_norm, 0.0, 1.0);
    double v_raw = v_max * std::pow(1.0 - kappa_norm, alpha);
    speeds[i] = std::max(v_raw, v_min);
  }

  // Step 4: Rolling-window smoothing
  if (smooth_window > 1 && n > 1) {
    const int window = std::min(smooth_window, static_cast<int>(n));
    const int half = window / 2;
    std::vector<double> smoothed = speeds;

    for (std::size_t i = 0; i < n; ++i) {
      double sum = 0.0;
      int count = 0;
      for (int j = -half; j <= half; ++j) {
        int idx = static_cast<int>(i) + j;
        if (idx < 0) idx += static_cast<int>(n);
        if (idx >= static_cast<int>(n)) idx -= static_cast<int>(n);
        sum += speeds[static_cast<std::size_t>(idx)];
        ++count;
      }
      smoothed[i] = sum / static_cast<double>(count);
    }
    speeds = smoothed;
  }

  // Step 5: Final clamp
  for (std::size_t i = 0; i < n; ++i) {
    speeds[i] = clampValue(speeds[i], v_min, v_max);
  }

  return speeds;
}

// ============================================================================
// Nearest-point projection on piecewise-linear trajectory
//
// trajectory: (2, N) matrix of [x; y] waypoints
// Returns: (nearest_point, distance, t, segment_index)
//   where t ∈ [0, 1] is the interpolation parameter along the segment
// ============================================================================
inline std::tuple<Eigen::Vector2d, double, double, std::size_t>
nearestPoint(const Eigen::Vector2d & point, const Eigen::MatrixXd & trajectory)
{
  int n = static_cast<int>(trajectory.cols());
  if (n <= 0) {
    return {Eigen::Vector2d::Zero(),
            std::numeric_limits<double>::infinity(), 0.0, 0};
  }
  if (n == 1) {
    return {trajectory.col(0), (trajectory.col(0) - point).norm(), 0.0, 0};
  }

  Eigen::Vector2d best_point = trajectory.col(0);
  double best_distance = std::numeric_limits<double>::infinity();
  double best_t = 0.0;
  std::size_t best_segment = 0U;

  for (int i = 0; i < n; ++i) {
    const int next = (i + 1) % n;
    const Eigen::Vector2d start = trajectory.col(i);
    const Eigen::Vector2d end = trajectory.col(next);
    const Eigen::Vector2d diff = end - start;
    const double l2 = std::max(diff.squaredNorm(), kNearZero);
    const double t = clampValue((point - start).dot(diff) / l2, 0.0, 1.0);
    const Eigen::Vector2d projection = start + t * diff;
    const double distance = (projection - point).norm();

    if (distance < best_distance) {
      best_distance = distance;
      best_point = projection;
      best_t = t;
      best_segment = static_cast<std::size_t>(i);
    }
  }

  return {best_point, best_distance, best_t, best_segment};
}

// ============================================================================
// Interpolated reference trajectory (discrete mode)
//
// Builds a (NX, horizon+1) reference trajectory by linear interpolation
// between waypoints, stepping forward by v_ref * DTK at each step.
//
// @param x, y      Current vehicle position
// @param cx, cy    Waypoint x/y coordinates (N,)
// @param cv        Waypoint reference speeds (N,) — already CiMPCC-mapped
// @param cyaw      Waypoint yaw angles (N,)
// @param dt        Time step
// @param horizon   Prediction horizon (returns horizon+1 columns)
// @return ref_traj (NX, horizon+1), rows: [x, y, v, yaw]
// ============================================================================
inline Eigen::MatrixXd calcInterpolatedRefTrajectory(
    double x, double y,
    const Eigen::VectorXd & cx,
    const Eigen::VectorXd & cy,
    const Eigen::VectorXd & cv,
    const Eigen::VectorXd & cyaw,
    double dt, int horizon)
{
  int ncourse = static_cast<int>(cx.size());
  Eigen::MatrixXd ref_traj = Eigen::MatrixXd::Zero(NX, horizon + 1);

  if (ncourse < 2) return ref_traj;

  // Waypoint spacing (assumed uniform)
  double dl = std::hypot(cx(1) - cx(0), cy(1) - cy(0));
  if (dl < kNearZero) dl = kNearZero;

  // Build (2, N) trajectory matrix for nearestPoint
  Eigen::MatrixXd trajectory(2, ncourse);
  trajectory.row(0) = cx.transpose();
  trajectory.row(1) = cy.transpose();

  // Find nearest point
  auto [nearest_pt, dist, t_current, ind_current] =
    nearestPoint(Eigen::Vector2d(x, y), trajectory);
  (void)nearest_pt; (void)dist;

  // Build t_list: accumulated progress in segment units
  Eigen::VectorXd t_list = Eigen::VectorXd::Zero(horizon + 1);
  t_list(0) = t_current;

  int ind_next_wp = (static_cast<int>(ind_current) + 1) % ncourse;
  double current_speed =
    (1.0 - t_current) * cv(ind_current) + t_current * cv(ind_next_wp);

  for (int i = 1; i <= horizon; ++i) {
    t_list(i) = t_list(i - 1) + (current_speed * dt) / dl;

    double t_frac = std::fmod(t_list(i), 1.0);
    if (t_frac < 0.0) t_frac += 1.0;

    int seg_idx = (static_cast<int>(std::floor(t_list(i))) +
                   static_cast<int>(ind_current)) % ncourse;
    if (seg_idx < 0) seg_idx += ncourse;
    int seg_next = (seg_idx + 1) % ncourse;

    current_speed = (1.0 - t_frac) * cv(seg_idx) + t_frac * cv(seg_next);
  }

  // Interpolate all state variables
  for (int i = 0; i <= horizon; ++i) {
    int idx = (static_cast<int>(std::floor(t_list(i))) +
               static_cast<int>(ind_current)) % ncourse;
    if (idx < 0) idx += ncourse;
    int idx_next = (idx + 1) % ncourse;

    double t_frac = std::fmod(t_list(i), 1.0);
    if (t_frac < 0.0) t_frac += 1.0;

    ref_traj(X, i) = (1.0 - t_frac) * cx(idx) + t_frac * cx(idx_next);
    ref_traj(Y, i) = (1.0 - t_frac) * cy(idx) + t_frac * cy(idx_next);
    ref_traj(V, i) = (1.0 - t_frac) * cv(idx) + t_frac * cv(idx_next);

    double yaw0 = cyaw(idx);
    double yaw1 = cyaw(idx_next);
    double yaw_diff = std::atan2(std::sin(yaw1 - yaw0),
                                 std::cos(yaw1 - yaw0));
    ref_traj(YAW, i) = yaw0 + t_frac * yaw_diff;
  }

  return ref_traj;
}

}  // namespace mpc_control

#endif  // MPC_CONTROL_MPC_UTILS_HPP_
