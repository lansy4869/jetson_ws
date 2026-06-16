# F1TENTH 部署优先的特权蒸馏竞速方案

本文档整理一条以实车部署为优先目标的 F1TENTH 竞速路线：将主骨架从残差强化学习切换为“特权专家 + 模仿蒸馏”，再用无状态 CNN、可测输入、3D 对手感知和 g-g-v 安全壳降低部署复杂度与 sim-to-real 风险。

## 1. 核心判断：决定好不好部署的不是网络，而是训练范式

读完 RaceMOP、α-RPO 和 End2Race 的关键代码后，一个会改变整体路线的判断是：

> 若目标是“更好部署”，最大的杠杆不是换一个更小的网络，而是把训练范式从残差 RL 换成模仿学习 / 蒸馏。

三套方法在部署友好度上的真实差距如下。

| 方法 | RaceMOP：残差 PPO | α-RPO：衰减残差 PPO | End2Race：纯模仿 BC |
| --- | --- | --- | --- |
| 训练 | 30M 步、奖励整形、PPO 不稳定 | 2.5M 步、仍是 RL，仍有奖励设计和 sim-to-real RL gap | 无 RL：采专家示范 + 100 epoch MSE |
| 部署时运行内容 | base APF 必须在线运行；`agents.py:142` 中为 `a * scale + base`；APF 是 JAX 20 步梯度下降 / 帧，Jetson 上重且难装 | 只跑残差网络 | 只跑一个网络 |
| 策略输入 | 包含 `slip_angle` / `yaw_rate`，实车难测 | 同左 | 仅 LiDAR + 速度，全部可测 |
| sim-to-real | RL gap 大 | RL gap 中 | 无 RL gap，行为克隆是确定性监督学习 |

结论：

End2Race 已经证明纯 BC 能在 F1TENTH 上做出超车，成功率约 59%。它没有奖励整形、没有 PPO、没有 RL 的 sim-to-real gap，部署时只跑一个网络，且输入全可测。部署友好度上，它比 α-RPO 还高一个量级。

因此，若论文和工程目标强调“好部署”，主线应调整为：

> 特权专家在仿真中产示范，学生网络通过 BC + DAgger 蒸馏，实车部署时只运行学生网络。

## 2. 两个仓库各有部署坑，应交叉取 best

不应整套照搬 End2Race 或 RaceMOP，而是分别取它们更部署友好的一部分。

| 部件 | 建议来源 | 部署角度理由 |
| --- | --- | --- |
| 训练范式：BC + DAgger | End2Race | 无 RL、无奖励整形、无 RL gap；DAgger 廉价修正 BC 的协变量漂移，仍不需要奖励函数 |
| 学生骨干：无状态帧堆叠 1D-CNN，约 826K 参数 | RaceMOP `nn.py` 中的 `LidarNet` 思路 | 不使用 End2Race 的 GRU；GRU 约 5.2M 参数且有状态，TensorRT 导出和部署更麻烦；CNN 无状态、TRT 友好，推理约 1-2 ms |
| 输入编码：可学习 sigmoid 压力 token | End2Race `model.py:81` | 逐元素操作廉价，突出近处障碍，对 LiDAR 噪声更鲁棒 |
| 学生观测：LiDAR、速度、`opponent_state` | End2Race 思路 | 全部实车可测；`slip_angle`、`yaw_rate` 只进入特权专家或 critic，不进入学生部署输入 |
| 专家：仅仿真中使用的特权规划器 | End2Race lattice / RaceMOP APF / Piccinini 类方法 | 专家可使用地图、赛车线、对手真值等特权信息；只在仿真产示范，学生蒸馏完成后丢弃 |
| 安全壳：g-g-v | 当前 sysid 工作 | 用实车标定的横纵向极限替换 RaceMOP 的硬编码物理假设 |

推荐的学生输入是：

```text
observation_student = {
  lidar_stack_2d,
  ego_speed,
  opponent_state
}
```

推荐的特权专家输入是：

```text
observation_expert = {
  map,
  raceline,
  ego_full_state,
  opponent_full_state,
  vehicle_limits_from_sysid
}
```

部署时只保留：

```text
policy_student(lidar_stack_2d, ego_speed, opponent_state) -> steering, speed
g-g-v safety shell -> safe steering, safe speed
```

## 3. sysid / g-g-v 对 RaceMOP 硬编码漏洞的补位

RaceMOP 的 APF 速度剖面中有如下硬编码：

```python
velocity_con = np.sqrt(
    0.8 * 9.81 / (np.tan(np.abs(steering) + 1e-5) / 0.33)
)
```

这里包含两个关键假设：

1. `0.8 * 9.81`：假设横向极限为 0.8g。
2. `0.33`：假设车辆轴距为 0.33 m。

这两个参数都不适合当前车辆。当前车辆轴距是 0.25 m，且横向极限应来自实车 sysid / g-g-v 标定，而不是手写常数。

建议替换为：

```python
wheelbase = 0.25
ay_max = ggv_limits.ay_max(v)

curvature = np.tan(np.abs(steering) + 1e-5) / wheelbase
velocity_con = np.sqrt(ay_max / np.maximum(curvature, 1e-6))
```

纵向速度变化也应加入实测加减速饱和，例如：

```python
accel_limit = 6.5
brake_limit = -6.5
```

并在速度命令更新时做：

```python
dv = np.clip(v_target - v_current, brake_limit * dt, accel_limit * dt)
v_safe = v_current + dv
```

这样，当前 sysid 工作不是孤立模块，而是直接为部署管线提供两个能力：

1. 按本车真实物理极限限速。
2. 用 g-g-v 安全壳约束学习策略输出。

这也构成 sim-to-real 章节的实证依据：RaceMOP / End2Race 使用的仿真车辆参数并不等同于当前实车。例如 RaceMOP 速度剖面默认轴距为 0.33 m，而当前车辆为 0.25 m；横向极限和速度增益也需要实车辨识。

## 4. 部署优先的创新主线

推荐将论文和工程主线重排为：

> 蒸馏式无地图反应式竞速：在 sysid 接地的仿真环境中训练一个特权专家，专家可使用地图、赛车线和对手真值；再用 BC + DAgger 将专家蒸馏成一个轻量、无状态、仅使用 LiDAR + 速度 + opponent_state 的学生网络。部署时只运行学生网络；3D MID-360 在感知层产出 opponent_state，g-g-v 模型作为输出安全壳。

部署友好性来自五点：

1. 无 RL gap：主体训练是 BC / DAgger，而不是 PPO。
2. 单网络前向：实车部署时不在线跑 APF / JAX / lattice search。
3. 无状态 CNN：比 GRU 更容易导出 ONNX / TensorRT。
4. 输入全可测：LiDAR、速度和对手状态均可由实车传感器获得。
5. 物理安全壳：g-g-v 约束学习策略输出，降低越界动作风险。

## 5. 诚实的新颖性与取舍

需要明确的是：

1. BC 超车不是全新概念，End2Race 已经做过。
2. 特权专家到学生策略的蒸馏，End2Race 也已经覆盖了一部分。
3. 纯复刻 End2Race 不构成足够新颖性。

本路线的新颖性应放在以下组合上：

1. 3D MID-360 对手感知：用 3D LiDAR 单帧测速估计 `opponent_state`，并输入蒸馏学生策略。
2. sysid 接地的可迁移性：用实车辨识参数校准仿真和安全约束，而不是沿用仓库中的默认车辆参数。
3. 部署优先消融：围绕“到底什么对可部署的实车迁移有用”做系统评估。

建议的消融实验包括：

| 消融问题 | 对比项 | 目标 |
| --- | --- | --- |
| BC 是否足够 | BC vs DAgger vs DAgger + 残差 RL polish | 判断 RL 是否真的值得引入 |
| 对手感知维度是否有用 | 2D 对手感知 vs 3D MID-360 对手感知 | 量化 3D 感知对超车和避障的增益 |
| sysid 是否有效 | 默认仿真参数 vs sysid 接地参数 | 证明车辆参数辨识对 sim-to-real 的贡献 |
| 安全壳是否必要 | 无 g-g-v 约束 vs 有 g-g-v 约束 | 评估安全、圈速和动作平滑性的折中 |
| 学生结构是否影响部署 | GRU 学生 vs 无状态 CNN 学生 | 比较推理时延、TensorRT 兼容性和性能 |

BC 的天花板通常是专家水平，因为学生很难系统性超过老师。若后续目标是超过专家，可以在 BC + DAgger 已经稳定后，再增加一层薄的残差 RL polish。

因此，α-RPO 更适合作为可选增强，而不是主训练路线：

```text
主菜：特权专家 + BC + DAgger
甜点：薄残差 RL polish
```

## 6. 分阶段路线

### Phase A：单车、最稳部署闭环

目标：先做最容易部署、最容易验证的版本。

内容：

1. 在 sysid 接地的仿真中构建特权专家。
2. 专家可使用地图、赛车线和完整车辆状态。
3. 采集专家示范数据。
4. 训练无状态帧堆叠 1D-CNN 学生。
5. 学生输入仅包含 LiDAR 和自车速度。
6. 导出 ONNX，再转 TensorRT。
7. 验证推理时延小于 20 ms。

特点：

```text
零 RL
零实车风险
部署只跑一个学生网络
```

### Phase B：加入 DAgger 与对手状态

目标：修正 BC 的协变量漂移，并进入双车 / 多车场景。

内容：

1. 在闭环仿真中运行学生策略。
2. 对学生访问到的状态重新查询专家动作。
3. 聚合数据集并迭代训练学生。
4. 在观测中加入 `opponent_state`。
5. 专家继续使用对手真值，学生只使用可部署估计量。

关键点：

```text
DAgger 修正分布偏移
仍不引入奖励函数
仍不需要 RL
```

### Phase C：3D 感知与 g-g-v 安全壳上车

目标：把部署护城河接入真实车辆。

内容：

1. 使用 MID-360 点云估计对手状态。
2. 用单帧测速或短时跟踪输出 `opponent_state`。
3. 将 `opponent_state` 输入学生策略。
4. 在策略输出后接入 g-g-v 安全壳。
5. 用实车 sysid 参数限制横向和纵向可行动作。

部署链路：

```text
MID-360 -> opponent_state
2D LiDAR / 投影 LiDAR -> lidar_stack
VESC / odom -> ego_speed
student_policy -> raw steering, raw speed
g-g-v safety shell -> safe steering, safe speed
```

### Phase D：可选残差 RL polish

目标：在已经可部署的 BC + DAgger 策略基础上，尝试超过专家天花板。

内容：

1. 冻结或半冻结学生策略主体。
2. 训练一个很薄的残差层。
3. 使用小步数 RL 做 polish，而不是重新训练完整策略。
4. 保留 g-g-v 安全壳。
5. 与 BC / DAgger 基线严格对比。

判断标准：

只有当残差 RL polish 在实车迁移、稳定性和部署复杂度上收益明确时，才纳入最终系统。否则它只作为论文消融结果，而不是实车主线。

## 7. 推荐写入原方案的结构调整

若要把该路线合并回 `F1TENTH_3D残差竞速_α-RPO方案.md`，建议新增一节：

```text
部署优先变体：特权蒸馏
```

并对原路线做如下调整：

1. 将“残差 RL / α-RPO”从主线调整为可选后处理。
2. 将“BC + DAgger 蒸馏”设为主训练路线。
3. 将 RaceMOP APF 仅作为仿真专家候选，而不是实车在线模块。
4. 将 End2Race 的 GRU 学生替换为 RaceMOP 风格无状态 1D-CNN。
5. 将 `slip_angle`、`yaw_rate` 等难测状态移出学生观测，只保留给专家、critic 或评估模块。
6. 将 g-g-v 安全壳作为部署章节的必要模块。
7. 在 sim-to-real 章节明确写入车辆参数错配：轴距、横向极限、纵向加减速和速度增益均需 sysid 接地。
## 8. 一句话总结

想要“好部署”，答案是把骨架从“残差 RL”换成“特权专家 + 模仿蒸馏”：End2Race 证明了 BC 超车可行，且无 RL gap、部署只跑一个网络；RaceMOP 的无状态 CNN 比 GRU 更适合 TensorRT；当前 sysid / g-g-v 工作负责接地仿真和约束输出；3D MID-360 对手感知则构成实车部署和论文创新的主要护城河。
