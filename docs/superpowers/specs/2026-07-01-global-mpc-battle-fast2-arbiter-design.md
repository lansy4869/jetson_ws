# Global MPC + Battle Fast2 Arbiter Design

## Goal

在不破坏当前 `win0630` 可实车运行基线的前提下，引入全局 MPC 实验链路。默认实车入口仍然使用 `battle_fast2`，实验入口使用 particle filter 定位和全局参考线跑 MPC，并让 `battle_fast2` 作为局部避障和安全兜底层。

## Non-Goals

- 不直接替换现有 `battle_fast2.launch.py` 的默认行为。
- 不让多个节点同时直接发布 `/drive`。
- 第一版不把完整 `frenet_planning -> frenet_mpc` 作为主链路。
- 第一版不重写 `battle_fast2_node` 的核心避障算法。

## Current Baseline

当前可跑基线是 `roboracer_china_2025` 包中的 `battle_fast2_node`：

```text
/scan -> battle_fast2_node -> /drive
```

该链路已经在实车上验证成功，应作为默认回退方案保留。

当前工作区中已有两个未跟踪实验目录：

- `src/slash_navigation/mpc_control`
- `src/slash_navigation/frenet_planning`

静态检查显示 `mpc_control` 依赖 `track_spline`、`csv_data`、`OsqpEigen`。`frenet_planning` 还涉及 `reactive_racing`。这些依赖在当前工作区中尚未补齐，因此第一阶段应优先集成较小的 `mpc_control` 链路。

## Recommended Architecture

第一版采用三段式控制架构：

```text
/pf/pose/odom + waypoint_csv
        -> global MPC
        -> /mpc/drive_nominal

/scan
        -> battle_fast2 safety mode
        -> /battle_fast2/drive_reactive

/mpc/drive_nominal + /battle_fast2/drive_reactive + health/risk checks
        -> drive_arbiter
        -> /drive
```

职责划分：

- Global MPC: 使用 particle filter 的 `/pf/pose/odom` 和全局 waypoint CSV，生成沿赛道行驶的名义控制。
- Battle Fast2: 使用 `/scan` 做局部避障和安全兜底，实验模式下不直接发布 `/drive`。
- Drive Arbiter: 唯一发布 `/drive` 的节点，根据健康状态和风险状态选择 MPC、Battle Fast2 或停车。

## Topic Contract

默认实车模式保持不变：

```text
ros2 launch roboracer_china_2025 battle_fast2.launch.py
```

实验模式使用新话题，避免控制源冲突：

- `mpc_control` 发布 `/mpc/drive_nominal`，不直接发布 `/drive`。
- `battle_fast2_node` 在实验 launch 中发布 `/battle_fast2/drive_reactive`，不直接发布 `/drive`。
- `drive_arbiter` 发布 `/drive`。

实验 launch 需要显式传入：

- `odom_topic`, 默认 `/pf/pose/odom`
- `scan_topic`, 默认 `/scan`
- `waypoint_csv`, 必须指向实车赛道参考线
- `drive_topic`, 固定由 arbiter 输出为 `/drive`

## Arbitration Rules

`drive_arbiter` 的第一版规则保持可解释和保守：

1. MPC 名义控制正常且 Battle Fast2 未触发高风险时，输出 `/mpc/drive_nominal`。
2. MPC 超时、MPC 指令非有限值、particle filter odom 超时或 frame 不匹配时，输出停车指令。
3. Battle Fast2 检测到局部高风险时，输出 `/battle_fast2/drive_reactive`。
4. Battle Fast2 指令超时或非有限值时，不允许用它接管，改为停车。
5. 仲裁器输出前统一做 steering 和 speed 限幅。

第一版高风险信号可以用保守实现：

- 如果暂时没有独立 risk topic，Battle Fast2 reactive 指令只在 MPC 不健康时接管。
- 后续可从 Battle Fast2 中抽出 `risk_state` 或 `front_clearance` topic，让局部障碍接管更主动。

## Launch Layout

新增实验 launch，而不是修改默认 launch：

- 保留 `launch/battle_fast2.launch.py`
- 新增 `launch/global_mpc_battle_arbiter.launch.py`

实验 launch 启动：

- `mpc_control`，输出 `/mpc/drive_nominal`
- `battle_fast2_node`，输出 `/battle_fast2/drive_reactive`
- `drive_arbiter`，输出 `/drive`

如果实验依赖尚未补齐，应允许默认 `battle_fast2` 构建和运行不受影响。实验包可以先通过文档和 launch 隔离，待依赖补齐后再纳入实车构建验证。

## Error Handling

必须优先保证实车安全：

- 任何输入超时都不能继续复用旧控制指令。
- 任一控制指令出现 NaN 或 Inf 时立即判为无效。
- `/pf/pose/odom` frame 与配置的 `global_frame` 不一致时，MPC 无效。
- `waypoint_csv` 不存在或解析失败时，实验 launch 应启动失败，不影响默认 launch。
- 仲裁器应定期打印当前控制源和切换原因，便于实车调试。

## Testing Plan

本地环境当前没有 `colcon`，代码实现后需要在 ROS2 环境或实车 Jetson 上验证：

1. 默认链路回归：`battle_fast2.launch.py` 仍可启动，并且仍只发布 `/drive`。
2. 实验构建：补齐 `track_spline`、`csv_data`、`OsqpEigen` 后，`mpc_control` 可编译。
3. Topic 验证：实验 launch 中只有 `drive_arbiter` 发布 `/drive`。
4. 超时验证：停止 `/pf/pose/odom` 后，arbiter 输出停车。
5. 非有限值验证：给 arbiter 注入 NaN 指令后，arbiter 输出停车。
6. 实车低速验证：先把 `max_speed` 限制在低速，再逐步提高速度。

实验入口命令：

```bash
ros2 launch roboracer_china_2025 global_mpc_battle_arbiter.launch.py waypoint_csv:=/absolute/path/to/track.csv
```

实车 topic 安全检查：

```bash
ros2 topic info /drive
```

期望结果：实验模式下只有 `drive_arbiter` 发布 `/drive`。`mpc_control` 应发布 `/mpc/drive_nominal`，`battle_fast2_node` 应发布 `/battle_fast2/drive_reactive`。

## Future Extension

完整 `frenet_planning` 可以作为第二阶段接入：

```text
/scan + /pf/pose/odom + centerline_csv
        -> frenet_local_planner
        -> frenet_mpc
        -> /frenet/drive_nominal
        -> drive_arbiter
        -> /drive
```

第二阶段开始前需要先补齐 `reactive_racing`、`track_spline`、`csv_data`，并明确 Frenet 局部规划与 Battle Fast2 的职责边界，避免两个局部规划器重复决策。
