# Local Reachability Supervised MPC Safety Shield Design

## Goal

将当前 `mpcbattle` 实验链路从“全局 MPC + Battle Fast2 reactive fallback”升级为“局部可达性监督的 MPC Safety Shield”。核心目标是在不破坏默认实车基线的前提下，让 Battle Fast2 的可达性结果不只作为备用控制命令，还作为实时局部安全监督信号，对 MPC 名义控制进行速度限幅、转角融合、接管或停车。

## Motivation

单纯的 fallback 结构创新性较弱：MPC 健康时直接执行，MPC 不健康时才切 Battle Fast2。这种结构没有利用 Battle Fast2 每帧从 LaserScan 计算出的局部可达性信息，也不能在“MPC 数值健康但局部前方风险升高”时提前干预。

Safety Shield 将 Battle Fast2 升级为局部安全监督器：

```text
mpc_control
  -> /mpc/drive_nominal

battle_fast2_node
  -> /battle_fast2/drive_reactive
  -> /battle_fast2/front_clearance_m
  -> /battle_fast2/risk_min_margin_m
  -> /battle_fast2/reactive_speed_limit_mps

drive_arbiter
  -> local reachability supervised Safety Shield
  -> /drive
```

## Non-Goals

- 不替换默认 `battle_fast2.launch.py` 的 `/scan -> battle_fast2_node -> /drive` 基线。
- 不重写 `mpc_control` 求解器。
- 不把 `frenet_planning` 纳入本阶段主链路。
- 不引入自定义 ROS message；第一版只使用 `std_msgs/Float32` 和现有 `AckermannDriveStamped`。
- 不在第一版实现复杂概率风险估计或学习模块。

## Existing Inputs

当前可复用数据来自 `reachability_core.select_reachable_gap()` 的 `ReachabilityResult`：

- `valid`: 是否存在可达候选弧线。
- `steer`: Battle Fast2 reactive 建议转角。
- `speed`: 局部可达性允许的安全速度。
- `free_distance`: 最优候选弧线可通行距离。
- `min_clearance`: 最优候选弧线最小净空。
- `unknown_ratio`: 候选弧线上未知扫描比例。
- `score`: 候选评分。
- `reason`: 无效原因。

这些量已经在 `battle_fast2_node.py` 的 `try_reachability_control()` 中获得，因此第一版只需要把诊断发布出来，并让 `drive_arbiter` 订阅。

## Topic Contract

新增 Battle Fast2 诊断 topic：

| Topic | Type | 含义 |
| --- | --- | --- |
| `/battle_fast2/front_clearance_m` | `std_msgs/Float32` | 沿当前最优可达弧线的可通行距离，第一版取 `free_distance`。 |
| `/battle_fast2/risk_min_margin_m` | `std_msgs/Float32` | 最优可达弧线最小净空，第一版取 `min_clearance`。 |
| `/battle_fast2/reactive_speed_limit_mps` | `std_msgs/Float32` | 局部可达性给出的速度上限，第一版取 `speed`；无可达弧线时为 `0.0`。 |

保留现有控制 topic：

| Topic | Type | 含义 |
| --- | --- | --- |
| `/mpc/drive_nominal` | `ackermann_msgs/AckermannDriveStamped` | MPC 名义命令。 |
| `/battle_fast2/drive_reactive` | `ackermann_msgs/AckermannDriveStamped` | Battle Fast2 reactive 命令。 |
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | Safety Shield 后的唯一底盘命令。 |

## Shield Risk States

第一版使用确定性阈值划分五个状态：

```text
GREEN  : MPC 直接通过
YELLOW : MPC 转角保留，速度被 reactive_speed_limit 限制
ORANGE : MPC 与 reactive 转角加权融合，速度取保守值
RED    : 切 Battle Fast2 reactive
BLACK  : 停车
```

建议默认阈值：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `shield_diag_timeout_s` | `0.2` | 诊断 topic 最大允许延迟。 |
| `shield_yellow_margin_m` | `0.55` | 低于该净空进入 YELLOW。 |
| `shield_orange_margin_m` | `0.35` | 低于该净空进入 ORANGE。 |
| `shield_red_margin_m` | `0.18` | 低于该净空进入 RED。 |
| `shield_black_clearance_m` | `0.30` | 前向可通行距离低于该值停车。 |
| `shield_orange_blend` | `0.55` | ORANGE 中 reactive 转角权重。 |

状态计算规则：

1. odom 不健康：BLACK。
2. Safety diagnostics 缺失或超时：不允许主动风险屏蔽；若 MPC 健康则退化为原 MPC 优先策略，若 MPC 不健康则按原 reactive fallback/stop 处理。
3. `front_clearance_m <= shield_black_clearance_m` 或诊断非有限：BLACK。
4. `risk_min_margin_m <= shield_red_margin_m`：RED。
5. `risk_min_margin_m <= shield_orange_margin_m`：ORANGE。
6. `risk_min_margin_m <= shield_yellow_margin_m` 或 `mpc.speed > reactive_speed_limit_mps`：YELLOW。
7. 其他情况：GREEN。

## Shield Output Rules

所有输出都先检查新鲜度、数值有限性，再做最终限幅。

### GREEN

条件：MPC 健康，诊断为 GREEN。

输出：限幅后的 MPC 命令。

### YELLOW

条件：MPC 健康，局部风险轻微升高。

输出：

```text
steering = mpc.steering
speed = min(mpc.speed, reactive_speed_limit)
```

若 `reactive_speed_limit` 非有限或小于 0，则视作诊断无效，转入 BLACK。

### ORANGE

条件：MPC 健康，局部风险中等升高，reactive 命令健康。

输出：

```text
steering = (1 - w) * mpc.steering + w * reactive.steering
speed = min(mpc.speed, reactive.speed, reactive_speed_limit)
```

其中 `w = shield_orange_blend`。若 reactive 命令不可用，降级为 YELLOW 的速度限幅；若速度限幅也不可用，BLACK。

### RED

条件：局部风险高，reactive 命令健康。

输出：限幅后的 reactive 命令。

若 reactive 命令不可用，BLACK。

### BLACK

输出停车：

```text
steering = 0
speed = 0
```

## ROS Node Changes

### `battle_fast2_node.py`

新增参数：

- `front_clearance_topic`, 默认 `/battle_fast2/front_clearance_m`
- `risk_min_margin_topic`, 默认 `/battle_fast2/risk_min_margin_m`
- `reactive_speed_limit_topic`, 默认 `/battle_fast2/reactive_speed_limit_mps`

新增 publisher：

- `front_clearance_pub`
- `risk_min_margin_pub`
- `reactive_speed_limit_pub`

在 `try_reachability_control()` 中：

- Reachability 正常：发布 `free_distance`、`min_clearance`、`speed`。
- Reachability invalid 且不 fallback：发布 `0.0`、`-1.0`、`0.0`，同时发布停车 reactive 命令。
- Reachability 异常且不 fallback：发布 `0.0`、`-1.0`、`0.0`，同时发布停车 reactive 命令。
- 若 `reachability_fallback_to_original=True` 且回退到原始算法，则本阶段不发布可靠 Shield 诊断，避免把原始启发式输出误当作可达性证据。

### `drive_arbiter_core.py`

新增纯 Python 数据结构：

- `SafetyDiagnostics`
- `ShieldConfig`

扩展 `ArbiterConfig`，或在函数签名中显式传入 `ShieldConfig`。为了保持兼容，推荐保留 `ArbiterConfig` 并新增可选 `shield_config` 与 `safety_diagnostics` 参数；调用方不传时行为与当前版本一致。

新增纯函数：

```python
arbitrate(..., safety_diagnostics=None, safety_health=None, shield_config=None)
```

该函数在有有效 Shield 输入时执行上述 GREEN/YELLOW/ORANGE/RED/BLACK 规则；没有 Shield 输入时保持当前 MPC 优先、reactive fallback、stop 的行为。

### `drive_arbiter_node.py`

新增订阅：

- `std_msgs/Float32` `front_clearance_topic`
- `std_msgs/Float32` `risk_min_margin_topic`
- `std_msgs/Float32` `reactive_speed_limit_topic`

诊断健康状态按三路 topic 的最近更新时间综合。三者都更新且未超时时才视为 Shield 诊断健康；任一缺失则 Shield 降级为原仲裁逻辑。

新增参数：

- `enable_safety_shield`, 默认 `true`
- `front_clearance_topic`
- `risk_min_margin_topic`
- `reactive_speed_limit_topic`
- `shield_diag_timeout_s`
- `shield_yellow_margin_m`
- `shield_orange_margin_m`
- `shield_red_margin_m`
- `shield_black_clearance_m`
- `shield_orange_blend`

### Launch

`global_mpc_battle_arbiter.launch.py` 显式连接诊断 topic 和阈值参数。默认 `battle_fast2.launch.py` 可以暴露诊断 topic 参数，但默认行为仍直接发布 `/drive`。

## Testing Plan

### Unit Tests

扩展 `test_drive_arbiter_core.py`：

- GREEN：MPC 健康且 margin 高，输出 MPC。
- YELLOW：margin 轻微降低或 MPC 速度高于 speed limit，输出 MPC 转角 + 限速。
- ORANGE：margin 中等降低，输出转角融合和保守速度。
- RED：margin 很低，输出 reactive。
- BLACK：front clearance 过低，输出 stop。
- Shield 诊断缺失：保持原有仲裁行为。
- Shield 诊断超时：保持原有仲裁行为。
- 非有限诊断：输出 stop。

扩展 Battle Fast2 轻量检查：

- `battle_fast2_node.py` 中存在三个诊断 publisher。
- launch 中存在诊断 topic 参数。

### Runtime Checks

低速实车或 rosbag 回放中检查：

```bash
ros2 topic echo /battle_fast2/front_clearance_m
ros2 topic echo /battle_fast2/risk_min_margin_m
ros2 topic echo /battle_fast2/reactive_speed_limit_mps
ros2 topic echo /drive
```

实验模式中 `/drive` 仍只能由 `drive_arbiter` 发布。

## Paper Changes

论文题目建议改为：

```text
面向 F1TENTH 实车竞速的局部可达性监督 MPC 安全屏蔽控制方法研究
```

章节主线调整：

- 第 1 章：贡献从 fallback 仲裁升级为局部可达性监督 Safety Shield。
- 第 3 章：输入建模增加 Shield 诊断 topic。
- 第 4 章：Battle Fast2 从 reactive fallback 扩展为可达性风险监督器。
- 第 5 章：MPC 仍为名义全局跟踪控制器。
- 第 6 章：Drive Arbiter 改为 Safety Shield Arbiter，重点写五档风险状态与输出规则。
- 第 7 章：实验增加 GREEN/YELLOW/ORANGE/RED/BLACK 状态覆盖、限速效果、融合接管和停车保护。
- 第 8 章：总结突出局部可达性监督对 MPC 的连续安全约束。

## Risks

- 阈值需要低速实车调参，默认值必须保守。
- Battle Fast2 的 `free_distance` 是沿最优 reactive 弧线的可通行距离，不等同于 MPC 预测轨迹的碰撞检查；论文表述必须称为“局部可达性监督”，不能夸大为严格 MPC 轨迹安全证明。
- ORANGE 融合会改变 MPC 转角，存在引入不连续的风险；需要输出限幅，并在实验中统计转角变化率。
- 如果 `reachability_fallback_to_original=True`，Shield 诊断不应被视为可靠。

## Acceptance Criteria

- 默认 `battle_fast2.launch.py` 仍可独立发布 `/drive`。
- 实验 `global_mpc_battle_arbiter.launch.py` 中仍只有 `drive_arbiter` 发布 `/drive`。
- Battle Fast2 在 reachability 模式下发布三路 Shield 诊断。
- Arbiter 单元测试覆盖 GREEN/YELLOW/ORANGE/RED/BLACK。
- 论文 README、摘要和第 1/4/6/7/8 章与 Safety Shield 主线一致。
