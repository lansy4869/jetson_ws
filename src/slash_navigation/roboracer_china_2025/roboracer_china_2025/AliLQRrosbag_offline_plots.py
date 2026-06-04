#!/usr/bin/env python3
# coding=utf-8
"""
Offline metrics plotter for battle_fast2 rosbag2 logs.

Expected topics (default, auto-compatible old/new):
  - /battle_fast2/behavior_state                 (std_msgs/UInt8)
  - /battle_fast2/control_mode                   (std_msgs/UInt8)
  - /battle_fast2/current_speed_mps              (std_msgs/Float32)
  - /battle_fast2/alilqr_solve_time_ms           (std_msgs/Float32)
  - /battle_fast2/alilqr_ok                      (std_msgs/Bool)
  - /battle_fast2/obs_distance_m                 (std_msgs/Float32)
  - /battle_fast2/obs_rel_speed_mps              (std_msgs/Float32)
  - /battle_fast2/obs_is_dynamic                 (std_msgs/Bool)
  - /battle_fast2/front_clearance_m              (std_msgs/Float32)
  - /battle_fast2/risk_min_margin_m              (std_msgs/Float32)
  - /battle_fast2/steering_cmd_rad               (std_msgs/Float32)
  - /battle_fast2/speed_cmd_mps                  (std_msgs/Float32)
  - /battle_fast2/alilqr_speed_error_mps         (std_msgs/Float32)
  - /battle_fast2/alilqr_accel_cmd_mps2          (std_msgs/Float32)
  - /battle_fast2/location_confidence            (std_msgs/Float32)
  - /battle_fast2/perception_confidence          (std_msgs/Float32)
  - /battle_fast2/control_confidence             (std_msgs/Float32)
  - /battle_fast2/ttc_s                          (std_msgs/Float32)
  - /battle_fast2/risk_speed_scale               (std_msgs/Float32)
  - /battle_fast2/adaptive_q_scale               (std_msgs/Float32)
  - /battle_fast2/adaptive_r_scale               (std_msgs/Float32)
  - /battle_fast2/behavior_state_switch_count    (std_msgs/UInt32)
  - /battle_fast2/mode_switch_count              (std_msgs/UInt32)
"""

import argparse
import glob
import math
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import rosbag2_py
except Exception:
    rosbag2_py = None
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


BEHAVIOR_NAME_MAP = {
    0: "CRUISE",
    1: "FOLLOW",
    2: "OVERTAKE_PREPARE",
    3: "OVERTAKE_EXECUTE",
    4: "RETURN_TO_RACELINE",
}

CONTROL_MODE_NAME_MAP = {
    0: "ALILQR",
    1: "MODE_1",
    2: "MODE_2",
    3: "MODE_3",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot battle_fast2 metrics from rosbag2 db3")
    parser.add_argument("--bag", required=True, help="Path to rosbag2 directory")
    parser.add_argument("--out", default="", help="Output directory for figures (default: <bag>/plots)")
    parser.add_argument("--prefix", default="battle_metrics", help="Output file prefix")
    parser.add_argument("--dpi", type=int, default=170, help="Figure dpi")
    parser.add_argument("--max-solve-ms", type=float, default=80.0, help="Upper clip for solve time plot")
    return parser.parse_args()


def _get_topic_type_map(reader: Any) -> Dict[str, str]:
    info_list = reader.get_all_topics_and_types()
    return {item.name: item.type for item in info_list}


def _ensure_out_dir(bag_dir: str, out_dir: str) -> str:
    final_dir = out_dir if out_dir else os.path.join(bag_dir, "plots")
    os.makedirs(final_dir, exist_ok=True)
    return final_dir


def _safe_name(topic: str) -> str:
    return topic.replace("/", "_").strip("_")


def _build_reader(bag_dir: str) -> Any:
    if rosbag2_py is None:
        raise RuntimeError("rosbag2_py is not available in current Python environment.")
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="", output_serialization_format="")
    reader.open(storage_options, converter_options)
    return reader


def _read_topics_rosbag2_py(
    bag_dir: str,
    targets: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    reader = _build_reader(bag_dir)
    type_map = _get_topic_type_map(reader)
    available = set(type_map.keys())

    valid_topics = [t for t in targets if t in available]
    if len(valid_topics) == 0:
        raise RuntimeError("None of the target topics are present in bag.")

    storage_filter = rosbag2_py.StorageFilter(topics=valid_topics)
    reader.set_filter(storage_filter)

    msg_classes: Dict[str, object] = {}
    for topic in valid_topics:
        msg_classes[topic] = get_message(type_map[topic])

    times: Dict[str, List[float]] = {t: [] for t in valid_topics}
    values: Dict[str, List[float]] = {t: [] for t in valid_topics}

    first_t: Optional[float] = None
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic not in msg_classes:
            continue
        t_sec = stamp * 1e-9
        if first_t is None:
            first_t = t_sec
        rel_t = t_sec - first_t
        msg = deserialize_message(data, msg_classes[topic])
        v = float(getattr(msg, "data"))
        times[topic].append(rel_t)
        values[topic].append(v)

    np_times = {k: np.array(v, dtype=np.float64) for k, v in times.items()}
    np_values = {k: np.array(v, dtype=np.float64) for k, v in values.items()}
    return np_times, np_values


def _read_topics_sqlite(
    bag_dir: str,
    targets: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    db_files = sorted(glob.glob(os.path.join(bag_dir, "*.db3")))
    if len(db_files) == 0:
        raise RuntimeError(f"No .db3 files found under bag directory: {bag_dir}")

    records: List[Tuple[int, str, str, bytes]] = []
    present_topics = set()

    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        try:
            cur = conn.cursor()
            rows = cur.execute("SELECT id, name, type FROM topics").fetchall()
            id_meta = {int(r[0]): (str(r[1]), str(r[2])) for r in rows}

            valid_ids = [tid for tid, (name, _typ) in id_meta.items() if name in targets]
            if len(valid_ids) == 0:
                continue

            present_topics.update(id_meta[tid][0] for tid in valid_ids)
            placeholders = ",".join(["?"] * len(valid_ids))
            q = (
                "SELECT topic_id, timestamp, data FROM messages "
                f"WHERE topic_id IN ({placeholders}) ORDER BY timestamp ASC"
            )
            for topic_id, stamp, data in cur.execute(q, valid_ids):
                name, typ = id_meta[int(topic_id)]
                records.append((int(stamp), name, typ, bytes(data)))
        finally:
            conn.close()

    if len(present_topics) == 0:
        raise RuntimeError("None of the target topics are present in bag.")
    if len(records) == 0:
        raise RuntimeError("Target topics exist but contain no messages.")

    records.sort(key=lambda x: x[0])
    first_t = records[0][0] * 1e-9

    msg_classes: Dict[str, Any] = {}
    for _, _, typ, _ in records:
        if typ not in msg_classes:
            msg_classes[typ] = get_message(typ)

    times: Dict[str, List[float]] = {t: [] for t in sorted(present_topics)}
    values: Dict[str, List[float]] = {t: [] for t in sorted(present_topics)}
    for stamp, topic, typ, data in records:
        rel_t = stamp * 1e-9 - first_t
        msg = deserialize_message(data, msg_classes[typ])
        v = float(getattr(msg, "data"))
        times[topic].append(rel_t)
        values[topic].append(v)

    np_times = {k: np.array(v, dtype=np.float64) for k, v in times.items()}
    np_values = {k: np.array(v, dtype=np.float64) for k, v in values.items()}
    return np_times, np_values


def _read_topics(
    bag_dir: str,
    targets: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if rosbag2_py is not None:
        try:
            return _read_topics_rosbag2_py(bag_dir, targets)
        except Exception as e:
            print(f"[WARN] rosbag2_py reader failed: {e}")
            print("[WARN] Falling back to sqlite3 reader...")
    return _read_topics_sqlite(bag_dir, targets)


def _pick_topic(times: Dict[str, np.ndarray], candidates: List[str]) -> Optional[str]:
    for t in candidates:
        if t in times and len(times[t]) > 0:
            return t
    return None


def _step_series(ax, t: np.ndarray, y: np.ndarray, label_map: Dict[int, str], title: str, ylabel: str):
    if len(t) == 0:
        ax.set_title(title + " (no data)")
        ax.grid(True, alpha=0.3)
        return
    yi = y.astype(np.int32)
    ax.step(t, yi, where="post", linewidth=1.5)
    ticks = sorted(list(set(yi.tolist())))
    tick_labels = [label_map.get(int(v), str(int(v))) for v in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def _save(fig, path: str, dpi: int):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_timeline(
    out_dir: str,
    prefix: str,
    dpi: int,
    times: Dict[str, np.ndarray],
    values: Dict[str, np.ndarray],
):
    behavior_topic = "/battle_fast2/behavior_state"
    mode_topic = "/battle_fast2/control_mode"
    speed_topic = "/battle_fast2/current_speed_mps"
    solve_topic = _pick_topic(times, ["/battle_fast2/alilqr_solve_time_ms", "/battle_fast2/mpc_solve_time_ms"])

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    _step_series(
        axes[0],
        times.get(behavior_topic, np.array([])),
        values.get(behavior_topic, np.array([])),
        BEHAVIOR_NAME_MAP,
        "Behavior State Timeline",
        "Behavior",
    )
    _step_series(
        axes[1],
        times.get(mode_topic, np.array([])),
        values.get(mode_topic, np.array([])),
        CONTROL_MODE_NAME_MAP,
        "Control Mode Timeline",
        "Mode",
    )

    t_speed = times.get(speed_topic, np.array([]))
    v_speed = values.get(speed_topic, np.array([]))
    if len(t_speed) > 0:
        axes[2].plot(t_speed, v_speed, linewidth=1.6, color="#1f77b4")
        axes[2].set_title("Vehicle Speed")
        axes[2].set_ylabel("m/s")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].set_title("Vehicle Speed (no data)")
        axes[2].grid(True, alpha=0.3)

    t_solve = times.get(solve_topic, np.array([])) if solve_topic is not None else np.array([])
    v_solve = values.get(solve_topic, np.array([])) if solve_topic is not None else np.array([])
    if len(t_solve) > 0:
        axes[3].plot(t_solve, v_solve, linewidth=1.2, color="#d62728")
        title_name = "AliLQR" if solve_topic and "alilqr" in solve_topic else "MPC"
        axes[3].set_title(f"{title_name} Solve Time")
        axes[3].set_ylabel("ms")
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].set_title("Solve Time (no data)")
        axes[3].grid(True, alpha=0.3)

    axes[3].set_xlabel("Time (s)")
    out_path = os.path.join(out_dir, f"{prefix}_timeline.png")
    _save(fig, out_path, dpi)
    return out_path


def _plot_distribution(
    out_dir: str,
    prefix: str,
    dpi: int,
    times: Dict[str, np.ndarray],
    values: Dict[str, np.ndarray],
):
    behavior_topic = "/battle_fast2/behavior_state"
    mode_topic = "/battle_fast2/control_mode"
    ok_topic = _pick_topic(times, ["/battle_fast2/alilqr_ok", "/battle_fast2/mpc_ok"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    def bar_for_discrete(ax, arr: np.ndarray, name_map: Dict[int, str], title: str):
        if len(arr) == 0:
            ax.set_title(title + " (no data)")
            ax.grid(True, axis="y", alpha=0.3)
            return
        ai = arr.astype(np.int32)
        keys, counts = np.unique(ai, return_counts=True)
        labels = [name_map.get(int(k), str(int(k))) for k in keys]
        perc = counts / np.sum(counts) * 100.0
        bars = ax.bar(labels, perc, color="#4e79a7")
        for b, p in zip(bars, perc):
            ax.text(b.get_x() + b.get_width() * 0.5, p + 0.8, f"{p:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("Ratio (%)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)

    bar_for_discrete(axes[0], values.get(behavior_topic, np.array([])), BEHAVIOR_NAME_MAP, "Behavior Occupancy")
    bar_for_discrete(axes[1], values.get(mode_topic, np.array([])), CONTROL_MODE_NAME_MAP, "Control Mode Occupancy")

    ok_arr = values.get(ok_topic, np.array([])) if ok_topic is not None else np.array([])
    if len(ok_arr) == 0:
        axes[2].set_title("Controller Success Ratio (no data)")
        axes[2].grid(True, axis="y", alpha=0.3)
    else:
        ok_ratio = float(np.mean(ok_arr > 0.5) * 100.0)
        fail_ratio = max(0.0, 100.0 - ok_ratio)
        bars = axes[2].bar(["OK", "Fail"], [ok_ratio, fail_ratio], color=["#59a14f", "#e15759"])
        for b in bars:
            y = b.get_height()
            axes[2].text(b.get_x() + b.get_width() * 0.5, y + 0.8, f"{y:.1f}%", ha="center", va="bottom", fontsize=10)
        title_name = "AliLQR" if ok_topic and "alilqr" in ok_topic else "MPC"
        axes[2].set_title(f"{title_name} Success Ratio")
        axes[2].set_ylabel("Ratio (%)")
        axes[2].grid(True, axis="y", alpha=0.3)

    out_path = os.path.join(out_dir, f"{prefix}_distribution.png")
    _save(fig, out_path, dpi)
    return out_path


def _plot_speed_solve_scatter(
    out_dir: str,
    prefix: str,
    dpi: int,
    max_solve_ms: float,
    times: Dict[str, np.ndarray],
    values: Dict[str, np.ndarray],
):
    speed_topic = "/battle_fast2/current_speed_mps"
    solve_topic = _pick_topic(times, ["/battle_fast2/alilqr_solve_time_ms", "/battle_fast2/mpc_solve_time_ms"])

    ts = times.get(speed_topic, np.array([]))
    vs = values.get(speed_topic, np.array([]))
    tm = times.get(solve_topic, np.array([])) if solve_topic is not None else np.array([])
    vm = values.get(solve_topic, np.array([])) if solve_topic is not None else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    if len(ts) > 1 and len(tm) > 1:
        vm_clipped = np.clip(vm, 0.0, max_solve_ms)
        solve_on_speed_time = np.interp(ts, tm, vm_clipped, left=vm_clipped[0], right=vm_clipped[-1])
        axes[0].scatter(vs, solve_on_speed_time, s=8, alpha=0.45, c="#f28e2b", edgecolors="none")
        axes[0].set_xlabel("Speed (m/s)")
        axes[0].set_ylabel("Solve Time (ms)")
        title_name = "AliLQR" if solve_topic and "alilqr" in solve_topic else "MPC"
        axes[0].set_title(f"Speed vs {title_name} Solve Time")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].set_title("Speed vs Solve Time (no paired data)")
        axes[0].grid(True, alpha=0.3)

    if len(vm) > 0:
        vm_clipped = np.clip(vm, 0.0, max_solve_ms)
        axes[1].hist(vm_clipped, bins=30, color="#76b7b2", alpha=0.9)
        p50 = float(np.percentile(vm_clipped, 50))
        p90 = float(np.percentile(vm_clipped, 90))
        p99 = float(np.percentile(vm_clipped, 99))
        axes[1].axvline(p50, color="#1f77b4", linestyle="--", linewidth=1.2, label=f"P50={p50:.2f}ms")
        axes[1].axvline(p90, color="#ff7f0e", linestyle="--", linewidth=1.2, label=f"P90={p90:.2f}ms")
        axes[1].axvline(p99, color="#d62728", linestyle="--", linewidth=1.2, label=f"P99={p99:.2f}ms")
        axes[1].set_xlabel("Solve Time (ms)")
        axes[1].set_ylabel("Count")
        title_name = "AliLQR" if solve_topic and "alilqr" in solve_topic else "MPC"
        axes[1].set_title(f"{title_name} Solve Time Distribution")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_title("Solve Time Distribution (no data)")
        axes[1].grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, f"{prefix}_solve_stats.png")
    _save(fig, out_path, dpi)
    return out_path


def _plot_safety_dynamic(
    out_dir: str,
    prefix: str,
    dpi: int,
    times: Dict[str, np.ndarray],
    values: Dict[str, np.ndarray],
):
    obs_dis_topic = "/battle_fast2/obs_distance_m"
    obs_rel_speed_topic = "/battle_fast2/obs_rel_speed_mps"
    obs_dynamic_topic = "/battle_fast2/obs_is_dynamic"
    front_clear_topic = "/battle_fast2/front_clearance_m"
    risk_margin_topic = "/battle_fast2/risk_min_margin_m"

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    t_dis = times.get(obs_dis_topic, np.array([]))
    v_dis = values.get(obs_dis_topic, np.array([]))
    if len(t_dis) > 0:
        valid = v_dis >= 0.0
        axes[0].plot(t_dis[valid], v_dis[valid], linewidth=1.4, color="#4e79a7", label="obs_distance")
        axes[0].set_ylabel("m")
        axes[0].set_title("Obstacle Distance")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].set_title("Obstacle Distance (no data)")
        axes[0].grid(True, alpha=0.3)

    t_rel = times.get(obs_rel_speed_topic, np.array([]))
    v_rel = values.get(obs_rel_speed_topic, np.array([]))
    if len(t_rel) > 0:
        axes[1].plot(t_rel, v_rel, linewidth=1.3, color="#f28e2b", label="obs_rel_speed")
        axes[1].axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        axes[1].set_ylabel("m/s")
        axes[1].set_title("Relative Obstacle Speed")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_title("Relative Obstacle Speed (no data)")
        axes[1].grid(True, alpha=0.3)

    _step_series(
        axes[2],
        times.get(obs_dynamic_topic, np.array([])),
        values.get(obs_dynamic_topic, np.array([])),
        {0: "STATIC", 1: "DYNAMIC"},
        "Dynamic Obstacle Flag",
        "Flag",
    )

    t_clear = times.get(front_clear_topic, np.array([]))
    v_clear = values.get(front_clear_topic, np.array([]))
    t_margin = times.get(risk_margin_topic, np.array([]))
    v_margin = values.get(risk_margin_topic, np.array([]))
    has_data_4th = False
    if len(t_clear) > 0:
        axes[3].plot(t_clear, v_clear, linewidth=1.4, color="#59a14f", label="front_clearance")
        has_data_4th = True
    if len(t_margin) > 0:
        valid_margin = v_margin >= -0.5
        axes[3].plot(t_margin[valid_margin], v_margin[valid_margin], linewidth=1.4, color="#e15759", label="risk_min_margin")
        has_data_4th = True
    if has_data_4th:
        axes[3].set_title("Front Clearance / Risk Margin")
        axes[3].set_ylabel("m")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].set_title("Front Clearance / Risk Margin (no data)")
        axes[3].grid(True, alpha=0.3)

    axes[3].set_xlabel("Time (s)")
    out_path = os.path.join(out_dir, f"{prefix}_safety_dynamic.png")
    _save(fig, out_path, dpi)
    return out_path


def _plot_control_cmds(
    out_dir: str,
    prefix: str,
    dpi: int,
    times: Dict[str, np.ndarray],
    values: Dict[str, np.ndarray],
):
    steer_topic = "/battle_fast2/steering_cmd_rad"
    speed_cmd_topic = "/battle_fast2/speed_cmd_mps"
    accel_cmd_topic = "/battle_fast2/alilqr_accel_cmd_mps2"

    fig, axes = plt.subplots(3, 1, figsize=(14, 8.8), sharex=True)

    ts = times.get(steer_topic, np.array([]))
    vs = values.get(steer_topic, np.array([]))
    if len(ts) > 0:
        axes[0].plot(ts, vs, linewidth=1.4, color="#af7aa1")
        axes[0].set_title("Steering Command")
        axes[0].set_ylabel("rad")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].set_title("Steering Command (no data)")
        axes[0].grid(True, alpha=0.3)

    tv = times.get(speed_cmd_topic, np.array([]))
    vv = values.get(speed_cmd_topic, np.array([]))
    if len(tv) > 0:
        axes[1].plot(tv, vv, linewidth=1.4, color="#76b7b2")
        axes[1].set_title("Speed Command")
        axes[1].set_ylabel("m/s")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_title("Speed Command (no data)")
        axes[1].grid(True, alpha=0.3)

    ta = times.get(accel_cmd_topic, np.array([]))
    va = values.get(accel_cmd_topic, np.array([]))
    if len(ta) > 0:
        axes[2].plot(ta, va, linewidth=1.4, color="#f28e2b")
        axes[2].axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        axes[2].set_title("Acceleration Command")
        axes[2].set_ylabel("m/s^2")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].set_title("Acceleration Command (no data)")
        axes[2].grid(True, alpha=0.3)

    axes[2].set_xlabel("Time (s)")
    out_path = os.path.join(out_dir, f"{prefix}_control_cmd.png")
    _save(fig, out_path, dpi)
    return out_path


def _plot_adaptive_risk_confidence(
    out_dir: str,
    prefix: str,
    dpi: int,
    times: Dict[str, np.ndarray],
    values: Dict[str, np.ndarray],
):
    loc_conf_topic = "/battle_fast2/location_confidence"
    percep_conf_topic = "/battle_fast2/perception_confidence"
    ctrl_conf_topic = "/battle_fast2/control_confidence"
    ttc_topic = "/battle_fast2/ttc_s"
    risk_scale_topic = "/battle_fast2/risk_speed_scale"
    q_scale_topic = "/battle_fast2/adaptive_q_scale"
    r_scale_topic = "/battle_fast2/adaptive_r_scale"
    speed_err_topic = "/battle_fast2/alilqr_speed_error_mps"

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    for topic, color, name in [
        (loc_conf_topic, "#4e79a7", "location_conf"),
        (percep_conf_topic, "#59a14f", "perception_conf"),
        (ctrl_conf_topic, "#f28e2b", "control_conf"),
    ]:
        t = times.get(topic, np.array([]))
        v = values.get(topic, np.array([]))
        if len(t) > 0:
            axes[0].plot(t, v, linewidth=1.3, color=color, label=name)
    if len(axes[0].lines) > 0:
        axes[0].set_title("Confidence Signals")
        axes[0].set_ylabel("0~1")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].set_title("Confidence Signals (no data)")
        axes[0].grid(True, alpha=0.3)

    t_ttc = times.get(ttc_topic, np.array([]))
    v_ttc = values.get(ttc_topic, np.array([]))
    t_rs = times.get(risk_scale_topic, np.array([]))
    v_rs = values.get(risk_scale_topic, np.array([]))
    has_risk = False
    if len(t_ttc) > 0:
        valid_ttc = v_ttc >= 0.0
        axes[1].plot(t_ttc[valid_ttc], v_ttc[valid_ttc], linewidth=1.3, color="#e15759", label="ttc_s")
        has_risk = True
    if len(t_rs) > 0:
        axes[1].plot(t_rs, v_rs, linewidth=1.3, color="#76b7b2", label="risk_speed_scale")
        has_risk = True
    if has_risk:
        axes[1].set_title("Risk Longitudinal Signals")
        axes[1].set_ylabel("value")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_title("Risk Longitudinal Signals (no data)")
        axes[1].grid(True, alpha=0.3)

    t_q = times.get(q_scale_topic, np.array([]))
    v_q = values.get(q_scale_topic, np.array([]))
    t_r = times.get(r_scale_topic, np.array([]))
    v_r = values.get(r_scale_topic, np.array([]))
    has_adapt = False
    if len(t_q) > 0:
        axes[2].plot(t_q, v_q, linewidth=1.3, color="#af7aa1", label="adaptive_q_scale")
        has_adapt = True
    if len(t_r) > 0:
        axes[2].plot(t_r, v_r, linewidth=1.3, color="#ff9da7", label="adaptive_r_scale")
        has_adapt = True
    if has_adapt:
        axes[2].set_title("Adaptive Gain Scheduling Scale")
        axes[2].set_ylabel("scale")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].set_title("Adaptive Gain Scheduling Scale (no data)")
        axes[2].grid(True, alpha=0.3)

    t_se = times.get(speed_err_topic, np.array([]))
    v_se = values.get(speed_err_topic, np.array([]))
    if len(t_se) > 0:
        axes[3].plot(t_se, v_se, linewidth=1.3, color="#9c755f")
        axes[3].axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        axes[3].set_title("AliLQR Speed Error")
        axes[3].set_ylabel("m/s")
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].set_title("AliLQR Speed Error (no data)")
        axes[3].grid(True, alpha=0.3)

    axes[3].set_xlabel("Time (s)")
    out_path = os.path.join(out_dir, f"{prefix}_adaptive_risk_conf.png")
    _save(fig, out_path, dpi)
    return out_path


def _print_summary(times: Dict[str, np.ndarray], values: Dict[str, np.ndarray]):
    speed_topic = "/battle_fast2/current_speed_mps"
    solve_topic = _pick_topic(times, ["/battle_fast2/alilqr_solve_time_ms", "/battle_fast2/mpc_solve_time_ms"])
    behavior_topic = "/battle_fast2/behavior_state"
    mode_topic = "/battle_fast2/control_mode"
    ok_topic = _pick_topic(times, ["/battle_fast2/alilqr_ok", "/battle_fast2/mpc_ok"])
    obs_dis_topic = "/battle_fast2/obs_distance_m"
    obs_dynamic_topic = "/battle_fast2/obs_is_dynamic"
    front_clear_topic = "/battle_fast2/front_clearance_m"
    risk_margin_topic = "/battle_fast2/risk_min_margin_m"
    ttc_topic = "/battle_fast2/ttc_s"
    conf_topic = "/battle_fast2/control_confidence"
    q_scale_topic = "/battle_fast2/adaptive_q_scale"
    r_scale_topic = "/battle_fast2/adaptive_r_scale"

    print("===== Summary =====")
    if len(times.get(speed_topic, [])) > 0:
        v = values[speed_topic]
        print(f"speed mean={np.mean(v):.3f} m/s, max={np.max(v):.3f} m/s")
    if solve_topic is not None and len(times.get(solve_topic, [])) > 0:
        s = values[solve_topic]
        title_name = "alilqr" if "alilqr" in solve_topic else "mpc"
        print(
            f"{title_name} solve ms "
            f"mean={np.mean(s):.3f}, p50={np.percentile(s, 50):.3f}, "
            f"p90={np.percentile(s, 90):.3f}, p99={np.percentile(s, 99):.3f}, max={np.max(s):.3f}"
        )
    if ok_topic is not None and len(values.get(ok_topic, [])) > 0:
        ok_ratio = np.mean(values[ok_topic] > 0.5) * 100.0
        title_name = "alilqr" if "alilqr" in ok_topic else "mpc"
        print(f"{title_name} success ratio={ok_ratio:.2f}%")
    if len(values.get(behavior_topic, [])) > 0:
        arr = values[behavior_topic].astype(np.int32)
        keys, counts = np.unique(arr, return_counts=True)
        desc = ", ".join([f"{BEHAVIOR_NAME_MAP.get(int(k), int(k))}:{c}" for k, c in zip(keys, counts)])
        print("behavior counts:", desc)
    if len(values.get(mode_topic, [])) > 0:
        arr = values[mode_topic].astype(np.int32)
        keys, counts = np.unique(arr, return_counts=True)
        desc = ", ".join([f"{CONTROL_MODE_NAME_MAP.get(int(k), int(k))}:{c}" for k, c in zip(keys, counts)])
        print("mode counts:", desc)
    if len(values.get(obs_dis_topic, [])) > 0:
        d = values[obs_dis_topic]
        valid = d[d >= 0.0]
        if len(valid) > 0:
            print(
                f"obs distance min={np.min(valid):.3f} m, mean={np.mean(valid):.3f} m, "
                f"p10={np.percentile(valid, 10):.3f} m"
            )
    if len(values.get(front_clear_topic, [])) > 0:
        fc = values[front_clear_topic]
        print(
            f"front clearance min={np.min(fc):.3f} m, mean={np.mean(fc):.3f} m, "
            f"p10={np.percentile(fc, 10):.3f} m"
        )
    if len(values.get(risk_margin_topic, [])) > 0:
        rm = values[risk_margin_topic]
        valid = rm[rm >= -0.5]
        if len(valid) > 0:
            print(
                f"risk margin min={np.min(valid):.3f} m, mean={np.mean(valid):.3f} m, "
                f"p10={np.percentile(valid, 10):.3f} m"
            )
    if len(values.get(obs_dynamic_topic, [])) > 0:
        dyn = values[obs_dynamic_topic] > 0.5
        print(f"dynamic-flag ratio={np.mean(dyn) * 100.0:.2f}%")
    if len(values.get(ttc_topic, [])) > 0:
        ttc = values[ttc_topic]
        valid = ttc[ttc >= 0.0]
        if len(valid) > 0:
            print(
                f"ttc min={np.min(valid):.3f} s, mean={np.mean(valid):.3f} s, "
                f"p10={np.percentile(valid, 10):.3f} s"
            )
    if len(values.get(conf_topic, [])) > 0:
        conf = values[conf_topic]
        print(
            f"control confidence mean={np.mean(conf):.3f}, "
            f"p10={np.percentile(conf, 10):.3f}, min={np.min(conf):.3f}"
        )
    if len(values.get(q_scale_topic, [])) > 0:
        qv = values[q_scale_topic]
        print(f"adaptive Q scale mean={np.mean(qv):.3f}, max={np.max(qv):.3f}")
    if len(values.get(r_scale_topic, [])) > 0:
        rv = values[r_scale_topic]
        print(f"adaptive R scale mean={np.mean(rv):.3f}, max={np.max(rv):.3f}")


def main():
    args = parse_args()
    bag_dir = os.path.abspath(args.bag)
    out_dir = _ensure_out_dir(bag_dir, args.out)

    targets = [
        "/battle_fast2/behavior_state",
        "/battle_fast2/control_mode",
        "/battle_fast2/current_speed_mps",
        "/battle_fast2/alilqr_solve_time_ms",
        "/battle_fast2/alilqr_ok",
        "/battle_fast2/mpc_solve_time_ms",   # backward compatibility
        "/battle_fast2/mpc_ok",              # backward compatibility
        "/battle_fast2/obs_distance_m",
        "/battle_fast2/obs_rel_speed_mps",
        "/battle_fast2/obs_is_dynamic",
        "/battle_fast2/front_clearance_m",
        "/battle_fast2/risk_min_margin_m",
        "/battle_fast2/steering_cmd_rad",
        "/battle_fast2/speed_cmd_mps",
        "/battle_fast2/alilqr_speed_error_mps",
        "/battle_fast2/alilqr_accel_cmd_mps2",
        "/battle_fast2/location_confidence",
        "/battle_fast2/perception_confidence",
        "/battle_fast2/control_confidence",
        "/battle_fast2/ttc_s",
        "/battle_fast2/risk_speed_scale",
        "/battle_fast2/adaptive_q_scale",
        "/battle_fast2/adaptive_r_scale",
        "/battle_fast2/behavior_state_switch_count",
        "/battle_fast2/mode_switch_count",
    ]

    times, values = _read_topics(bag_dir, targets)
    p1 = _plot_timeline(out_dir, args.prefix, args.dpi, times, values)
    p2 = _plot_distribution(out_dir, args.prefix, args.dpi, times, values)
    p3 = _plot_speed_solve_scatter(out_dir, args.prefix, args.dpi, args.max_solve_ms, times, values)
    p4 = _plot_safety_dynamic(out_dir, args.prefix, args.dpi, times, values)
    p5 = _plot_control_cmds(out_dir, args.prefix, args.dpi, times, values)
    p6 = _plot_adaptive_risk_confidence(out_dir, args.prefix, args.dpi, times, values)
    _print_summary(times, values)

    print("===== Saved Figures =====")
    print(p1)
    print(p2)
    print(p3)
    print(p4)
    print(p5)
    print(p6)


if __name__ == "__main__":
    main()
