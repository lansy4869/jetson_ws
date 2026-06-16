#!/usr/bin/env python3
# coding=utf-8
"""
底盘系统辨识 + g-g-v 标定（F1TENTH / VESC）
读取 chassis_response_test（纵向）与 chassis_test（转向）两个 rosbag2，辨识：
  1) 纵向一阶+延迟模型：commanded speed -> measured speed 的 K / tau / Td
  2) 纵向加速度能力 |ax|_max（加速 / 制动）—— g-g-v 的纵轴（实测）
  3) 转向 setpoint 范围（servo duty -> 转角），并诚实标注低速下转向增益不可辨识
  4) g-g-v 可行域 + 限速 profile v_max(kappa)（纵轴实测、横轴按 mu 参数化）

只依赖 numpy + matplotlib(Agg) + rosbags（纯 python，不需装 ROS）。
用法： python3 chassis_sysid.py [--bags DIR] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# 来自 f1tenth_stack/config/vesc.yaml（这台车的真实标定）
SPEED_TO_ERPM_GAIN = 4614.0
STEER_ANGLE_TO_SERVO_GAIN = -1.2135
STEER_ANGLE_TO_SERVO_OFFSET = 0.5304
G = 9.81

TS = get_typestore(Stores.ROS2_FOXY)


def load_bag(path, topics):
    """读取指定话题，返回 {topic: np.ndarray}。Float64 -> (t,val)；Odometry -> (t,vx,wz)。"""
    out = {t: [] for t in topics}
    with AnyReader([Path(path)], default_typestore=TS) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for con, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, con.msgtype)
            t = t_ns / 1e9
            if con.msgtype == "std_msgs/msg/Float64":
                out[con.topic].append((t, float(msg.data)))
            elif con.msgtype == "nav_msgs/msg/Odometry":
                out[con.topic].append(
                    (t, float(msg.twist.twist.linear.x), float(msg.twist.twist.angular.z))
                )
    return {k: np.asarray(v) for k, v in out.items() if len(v)}


def resample(t, y, grid):
    return np.interp(grid, t, y)


def extract_step_segments(u, dt, min_dur=0.7, min_level=0.3, max_std=0.2, pre=0.5):
    """把指令切成分段常值，挑出"持续阶跃"段（保留前 pre 秒以含上升沿），返回 slice 列表。"""
    cps = [0] + list(np.where(np.abs(np.diff(u)) > 0.25)[0] + 1) + [len(u)]
    segs = []
    pre_n = int(pre / dt)
    for a, b in zip(cps[:-1], cps[1:]):
        if (b - a) * dt < min_dur:
            continue
        level = np.median(u[a:b])
        if abs(level) < min_level or np.std(u[a:b]) > max_std:
            continue
        segs.append(slice(max(0, a - pre_n), b))
    return segs


def _unit_response(ud, a):
    vr = np.zeros(len(ud))
    for k in range(len(ud) - 1):
        vr[k + 1] = vr[k] + a * (ud[k] - vr[k])
    return vr


def fit_first_order_delay(u, y, dt, tau_grid, td_grid, segs=None):
    """模型 v'=(K*u(t-Td)-v)/tau。只在阶跃段拟合（每段用各自初速度），(tau,Td) 网格 + 解析最优 K。"""
    n = len(u)
    if segs is None:
        segs = [slice(0, n)]
    best = None
    for td in td_grid:
        shift = int(round(td / dt))
        ud_full = np.concatenate([np.zeros(shift), u[: n - shift]]) if shift > 0 else u.copy()
        for tau in tau_grid:
            a = dt / tau
            num = den = 0.0
            ss_res = ss_tot = 0.0
            ymean = np.mean(np.concatenate([y[s] for s in segs]))
            pieces = []
            for s in segs:
                ud = ud_full[s]
                ys = y[s]
                v0 = ys[0]
                decay = (1 - a) ** np.arange(len(ys))      # 初速度自由衰减项
                vr = _unit_response(ud, a)                  # 单位增益受迫响应（零初值）
                base = v0 * decay
                num += float(vr @ (ys - base))
                den += float(vr @ vr)
                pieces.append((s, base, vr, ys))
            if den < 1e-9:
                continue
            K = num / den
            for s, base, vr, ys in pieces:
                fit = base + K * vr
                ss_res += float(((ys - fit) ** 2).sum())
                ss_tot += float(((ys - ymean) ** 2).sum())
            rmse = float(np.sqrt(ss_res / sum(s.stop - s.start for s in segs)))
            if best is None or rmse < best["rmse"]:
                best = {"K": float(K), "tau": float(tau), "Td": float(td),
                        "rmse": rmse, "r2": 1.0 - ss_res / (ss_tot or 1.0),
                        "segs": segs}
    # 生成全程拟合曲线（连续仿真，便于画图）
    shift = int(round(best["Td"] / dt))
    ud_full = np.concatenate([np.zeros(shift), u[: n - shift]]) if shift > 0 else u.copy()
    best["v_fit"] = best["K"] * _unit_response(ud_full, dt / best["tau"])
    return best


def accel_limits(v, dt, smooth_win=7, pct=99.0):
    """平滑速度后求 dv/dt 的分位数，作为加速 / 制动能力（剔除离群）。"""
    if smooth_win % 2 == 0:
        smooth_win += 1
    pad = smooth_win // 2
    kernel = np.ones(smooth_win) / smooth_win
    vs = np.convolve(np.pad(v, pad, mode="edge"), kernel, mode="valid")
    a = np.gradient(vs, dt)
    a_pos = a[a > 0]
    a_neg = a[a < 0]
    return {
        "ax_accel_max": float(np.percentile(a_pos, pct)) if len(a_pos) else 0.0,
        "ax_brake_max": float(-np.percentile(-a_neg, pct)) if len(a_neg) else 0.0,
        "accel_series": a, "v_smooth": vs,
    }


# ==================== 横向 / g-g-v 实测（绕圆 bag，W2 新增） ====================
def register_vesc_types(msg_dir):
    """把 vesc_msgs 自定义类型注册进全局 typestore TS，使 rosbags 能反序列化 /sensors/imu。"""
    try:
        from rosbags.typesys import get_types_from_msg
    except Exception:
        return False
    md = Path(msg_dir)
    candidates = {
        "vesc_msgs/msg/VescImu": "VescImu.msg",
        "vesc_msgs/msg/VescImuStamped": "VescImuStamped.msg",
        "vesc_msgs/msg/VescState": "VescState.msg",
        "vesc_msgs/msg/VescStateStamped": "VescStateStamped.msg",
    }
    types = {}
    for name, fn in candidates.items():
        p = md / fn
        if p.exists():
            try:
                types.update(get_types_from_msg(p.read_text(), name))
            except Exception:
                pass
    if not types:
        return False
    try:
        TS.register(types)
        return True
    except Exception:
        return False


def load_circle_bag(path):
    """读绕圆 bag：返回 odom(t,vx,wz)、servo(t,duty)、imu(t,ax,ay,az,gz)（imu 可能缺）。"""
    topics = ["/odom", "/commands/servo/position", "/sensors/imu"]
    odom, servo, imu = [], [], []
    with AnyReader([Path(path)], default_typestore=TS) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for con, t_ns, raw in reader.messages(connections=conns):
            t = t_ns / 1e9
            try:
                msg = reader.deserialize(raw, con.msgtype)
            except Exception:
                continue
            if con.topic == "/odom":
                odom.append((t, float(msg.twist.twist.linear.x),
                             float(msg.twist.twist.angular.z)))
            elif con.topic == "/commands/servo/position":
                servo.append((t, float(msg.data)))
            elif con.topic == "/sensors/imu":
                la = msg.imu.linear_acceleration
                imu.append((t, float(la.x), float(la.y), float(la.z),
                            float(msg.imu.angular_velocity.z)))
    return (np.asarray(odom) if odom else None,
            np.asarray(servo) if servo else None,
            np.asarray(imu) if imu else None)


def identify_lateral(path, dt, imu_units="auto", v_min_corner=0.5):
    """从绕圆 bag 实测横向极限 ay_max、有效轴距 L_eff、mu_eff。

    稳态自行车模型：横摆角速度 r = v·tan(δ)/L → L_eff = v·tan(δ)/r；
    横向加速度 a_y = v·r（IMU 直接实测，不依赖几何假设）。
    返回 (结果dict, 实测 ay_max 或 None)。
    """
    odom, servo, imu = load_circle_bag(path)
    res = {"source_bag": str(path)}
    if odom is None or servo is None:
        res.update({"identifiable": False,
                    "note": "绕圆 bag 缺 /odom 或 /commands/servo/position"})
        return res, None

    t0 = max(odom[0, 0], servo[0, 0])
    t1 = min(odom[-1, 0], servo[-1, 0])
    T = np.arange(0, max(dt, t1 - t0), dt)
    v = resample(odom[:, 0] - t0, odom[:, 1], T)
    duty = resample(servo[:, 0] - t0, servo[:, 1], T)
    delta = (duty - STEER_ANGLE_TO_SERVO_OFFSET) / STEER_ANGLE_TO_SERVO_GAIN  # rad

    corner = (np.abs(delta) > np.deg2rad(3.0)) & (v > v_min_corner)
    res["n_corner_samples"] = int(corner.sum())
    res["v_corner_max_mps"] = float(np.max(v[corner])) if corner.any() else float(np.max(v))

    ay_meas_max = None
    if imu is not None and len(imu) > 5:
        ay = resample(imu[:, 0] - t0, imu[:, 2], T)   # linear_acceleration.y
        gz = resample(imu[:, 0] - t0, imu[:, 4], T)   # angular_velocity.z
        units = imu_units
        if units == "auto":
            units = "g" if np.percentile(np.abs(ay), 95) < 3.0 else "mps2"
        ay = ay * (G if units == "g" else 1.0)
        res["imu_accel_units_used"] = units
        if corner.sum() > 10:
            ay_meas_max = float(np.percentile(np.abs(ay[corner]), 95))
            res["ay_lat_max_mps2_MEASURED"] = ay_meas_max
            res["yawrate_corner_max_radps"] = float(np.percentile(np.abs(gz[corner]), 95))
            # 有效轴距（仅在横摆显著的样本上估）
            gz_c, v_c, d_c = gz[corner], v[corner], delta[corner]
            good = np.abs(gz_c) > 0.05
            if good.sum() > 10:
                L_samples = v_c[good] * np.tan(d_c[good]) / gz_c[good]
                res["wheelbase_eff_m_MEASURED"] = float(np.median(np.abs(L_samples)))
    else:
        res["note_imu"] = "无 /sensors/imu 或样本过少：无法实测横向加速度（请确认录包含该话题）"

    identifiable = (ay_meas_max is not None and corner.sum() > 10
                    and res["v_corner_max_mps"] > v_min_corner)
    res["identifiable"] = bool(identifiable)
    res["mu_eff_MEASURED"] = float(ay_meas_max / G) if ay_meas_max else None
    if not identifiable:
        res["note"] = ("绕圆数据不足/无 IMU：未能实测横向极限。"
                       "确保绕圆时车速>0.5m/s、转角持续、且录了 /sensors/imu。")
    return res, ay_meas_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", default=str(Path(__file__).parent / "bags"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    ap.add_argument("--mu", type=float, default=1.0,
                    help="假设的轮胎-地面摩擦系数（无绕圆 bag 时横向按 mu 参数化）")
    ap.add_argument("--vmax", type=float, default=2.0, help="配置里的最大车速 m/s")
    ap.add_argument("--circle_bag", default=None,
                    help="绕圆/定半径 rosbag 目录；提供后用 VESC IMU 实测横向极限 ay_max（W2 核心）")
    ap.add_argument("--imu_accel_units", default="auto", choices=["auto", "g", "mps2"],
                    help="VESC IMU 线加速度单位（多数固件为 g，auto 自动判定）")
    ap.add_argument("--vesc_msg_dir",
                    default="src/slash_hardware/f1tenth_system/vesc/vesc_msgs/msg",
                    help="vesc_msgs/msg 目录（注册 VescImuStamped 类型，相对 jetson_ws 根）")
    args = ap.parse_args()
    bags = Path(args.bags)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fs = 50.0
    dt = 1.0 / fs
    results = {"calibration_in": {
        "speed_to_erpm_gain": SPEED_TO_ERPM_GAIN,
        "steering_angle_to_servo_gain": STEER_ANGLE_TO_SERVO_GAIN,
        "steering_angle_to_servo_offset": STEER_ANGLE_TO_SERVO_OFFSET}}

    # ---------- 1) 纵向辨识 ----------
    d = load_bag(bags / "chassis_response_test",
                 ["/commands/motor/speed", "/odom"])
    mot, od = d["/commands/motor/speed"], d["/odom"]
    t0 = min(mot[0, 0], od[0, 0])
    T = np.arange(0, min(mot[-1, 0], od[-1, 0]) - t0, dt)
    u = resample(mot[:, 0] - t0, mot[:, 1] / SPEED_TO_ERPM_GAIN, T)  # 指令速度 m/s
    v = resample(od[:, 0] - t0, od[:, 1], T)                         # 实测速度 m/s
    act = np.where((np.abs(u) > 0.05) | (np.abs(v) > 0.05))[0]
    sl = slice(act[0], act[-1] + 1)
    uA, vA, TA = u[sl], v[sl], T[sl]

    step_segs = extract_step_segments(uA, dt)
    fo = fit_first_order_delay(
        uA, vA, dt,
        tau_grid=np.linspace(0.04, 0.6, 40),
        td_grid=np.arange(0.0, 0.26, 0.01),
        segs=step_segs)
    al = accel_limits(vA, dt)
    # 稳态增益：指令与实测都明显非零时的稳健比例（中位数）
    mask = (np.abs(uA) > 0.4) & (np.abs(vA) > 0.4)
    ss_gain = float(np.median(vA[mask] / uA[mask])) if mask.sum() else float("nan")

    results["longitudinal"] = {
        "model": "v_dot = (K*u(t-Td) - v)/tau   (u=指令速度 m/s, v=实测速度 m/s)",
        "K": fo["K"], "tau_s": fo["tau"], "Td_s": fo["Td"],
        "fit_rmse_mps": fo["rmse"], "fit_r2": fo["r2"],
        "steady_state_gain_meas_over_cmd": ss_gain,
        "ax_accel_max_mps2": al["ax_accel_max"],
        "ax_brake_max_mps2": al["ax_brake_max"],
        "v_observed_max_mps": float(np.max(np.abs(vA))),
        "excitation_active_s": float(TA[-1] - TA[0]),
        "n_step_segments_used": len(step_segs),
        "note": "K/tau/Td 仅在持续阶跃段拟合（线性区）；短脉冲饱和区只用于求 ax_max。"
                "稳态增益≈0.93 => speed_to_erpm_gain=4614 基本成立（约7%稳态跟踪损失，载荷/滑移）；"
                "vmax 未被激励，沿用配置值",
    }

    # ---------- 2) 转向（仅 setpoint 范围，诚实标注） ----------
    ds = load_bag(bags / "chassis_test",
                  ["/commands/servo/position", "/odom"])
    servo = ds["/commands/servo/position"]
    od2 = ds["/odom"]
    servo_min, servo_max = float(servo[:, 1].min()), float(servo[:, 1].max())
    # servo duty -> 转角： angle = (servo - offset)/gain
    ang_at = lambda s: (s - STEER_ANGLE_TO_SERVO_OFFSET) / STEER_ANGLE_TO_SERVO_GAIN
    steer_hi = ang_at(servo_min)   # gain 为负，duty 小 -> 转角大（左/右取决符号）
    steer_lo = ang_at(servo_max)
    results["steering"] = {
        "servo_duty_observed": [servo_min, servo_max],
        "steering_angle_span_rad": sorted([float(steer_lo), float(steer_hi)]),
        "v_during_steer_max_mps": float(np.abs(od2[:, 1]).max()),
        "yawrate_during_steer_max_radps": float(np.abs(od2[:, 2]).max()),
        "identifiable": False,
        "note": "转向测试时车速≤0.31m/s（近静止），yaw≈0 => 转向->横摆增益与有效轴距不可辨识；"
                "只能给出 servo->转角的标定映射。需补一个有速度的转向/绕圆 bag。",
    }

    # ---------- 2.5) 横向（绕圆 bag，实测 ay_max；W2 新增） ----------
    measured_ay = None
    if args.circle_bag:
        if register_vesc_types(args.vesc_msg_dir):
            print("[lateral] 已注册 vesc_msgs 类型，可读 /sensors/imu")
        else:
            print("[lateral] 警告：未能注册 vesc_msgs 类型，IMU 横向加速度可能读不到")
        lat, measured_ay = identify_lateral(
            args.circle_bag, dt, imu_units=args.imu_accel_units)
        results["lateral"] = lat

    # ---------- 3) g-g-v 可行域 + 限速 profile ----------
    ax_max = max(abs(al["ax_accel_max"]), abs(al["ax_brake_max"]))  # 取幅值
    if measured_ay is not None:
        ay_max = measured_ay
        ay_source = "MEASURED (circle bag IMU)"
    else:
        ay_max = args.mu * G  # 横向极限：无绕圆 bag 时按 mu 参数化
        ay_source = f"ASSUMED (mu={args.mu})"
    kappa = np.linspace(0.02, 4.0, 200)  # 曲率 1/m
    v_curve = np.sqrt(ay_max / kappa)
    v_curve = np.clip(v_curve, 0, args.vmax)
    results["g_g_v"] = {
        "ax_long_max_mps2_MEASURED": float(ax_max),
        "ay_lat_max_mps2": float(ay_max),
        "ay_source": ay_source,
        "vmax_mps": args.vmax,
        "speed_profile_formula": "v_max(kappa) = min(vmax, sqrt(ay_max/kappa))",
        "note": "把本 json 的 ay_lat_max / ax_long_max 填进 "
                "slash_safety/config/ggv_shield.yaml（ay 实测优先，否则用 mu*g 假设并标注）。",
    }

    # ---------- 绘图 ----------
    fig, axs = plt.subplots(3, 1, figsize=(11, 11))
    axs[0].plot(TA, uA, label="commanded speed (ERPM/gain)", lw=1.2)
    axs[0].plot(TA, vA, label="measured speed (odom)", lw=1.2)
    axs[0].plot(TA, fo["v_fit"], "--", label=f"1st-order fit τ={fo['tau']:.3f}s Td={fo['Td']:.2f}s", lw=1.4)
    # 阶跃拟合段高亮
    for s in step_segs:
        axs[0].axvspan(TA[s.start], TA[min(s.stop, len(TA)) - 1], color="orange", alpha=0.12)
    axs[0].set_title(f"Longitudinal ID  K={fo['K']:.3f}  tau={fo['tau']:.3f}s  Td={fo['Td']:.2f}s  R2={fo['r2']:.3f}")
    axs[0].set_ylabel("speed [m/s]"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)

    axs[1].plot(TA, al["accel_series"], lw=0.8, color="tab:red")
    axs[1].axhline(al["ax_accel_max"], ls="--", color="g", label=f"+ax_max={al['ax_accel_max']:.2f}")
    axs[1].axhline(al["ax_brake_max"], ls="--", color="b", label=f"-ax_max={al['ax_brake_max']:.2f}")
    axs[1].set_title("Longitudinal accel dv/dt (smoothed)"); axs[1].set_ylabel("ax [m/s^2]")
    axs[1].set_xlabel("t [s]"); axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)

    # g-g-v ellipse + measured longitudinal points
    th = np.linspace(0, 2 * np.pi, 200)
    axs[2].plot(ay_max * np.cos(th), ax_max * np.sin(th), label="g-g ellipse (ay assumed)")
    axs[2].scatter([0, 0], [al["ax_accel_max"], -al["ax_brake_max"]],
                   color="r", zorder=5, label="measured longitudinal limits")
    axs[2].set_aspect("equal"); axs[2].set_xlabel("a_y [m/s^2]"); axs[2].set_ylabel("a_x [m/s^2]")
    axs[2].set_title(f"g-g-v (long=measured, lat=assumed mu={args.mu})"); axs[2].legend(fontsize=8); axs[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "sysid_overview.png", dpi=110)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(kappa, v_curve)
    ax2.set_xlabel("curvature kappa [1/m]"); ax2.set_ylabel("v_max [m/s]")
    ax2.set_title(f"Speed profile v_max(kappa)=min({args.vmax}, sqrt(ay_max/kappa))  ay_max={ay_max:.1f}")
    ax2.grid(alpha=.3); fig2.tight_layout(); fig2.savefig(out / "speed_profile.png", dpi=110)

    (out / "identified_params.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n[plots] {out/'sysid_overview.png'} , {out/'speed_profile.png'}")


if __name__ == "__main__":
    main()
