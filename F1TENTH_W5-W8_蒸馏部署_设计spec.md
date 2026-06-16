# F1TENTH W5–W8 设计 spec：sysid 接地仿真 + 特权蒸馏 + 部署 + 消融

> 本文是 [F1TENTH_部署时间安排.md](F1TENTH_部署时间安排.md) 中 **W5–W8（创新点 4 加分层 + §5 消融）** 的落地设计。
> W1–W4 已交付：`slash_safety`（g-g-v 护盾）、`slash_sysid`（底盘辨识+绕圆）、`multilayer_scan`（`/scan_3d`）、`obstacle_tracker`（`/perception/opponent_state`）。
> W5–W8 全部新增在单一包 **`src/slash_distill/`**（ament_python，离线训练 + ROS2 部署同包）。

---

## 0. 一句话

在 **sysid 接地** 的 `f110_gym` 里，用**中心线 pure-pursuit + g-g-v 限速** 的**特权专家**产示范，
用 **BC + DAgger** 蒸馏成一个**无状态 1D-CNN 学生**；部署时只跑学生（ONNX→TensorRT），
**学生不可信/超时 → 反应式兜底**，**输出过 g-g-v 护盾**；最后跑 §5 消融矩阵。

---

## 1. 已确认的设计决策

| 项 | 决定 |
|---|---|
| 包结构 | 单一 `src/slash_distill/`（离线 `python -m slash_distill.xxx` + ROS2 部署节点 + launch） |
| 特权专家 | 中心线 pure-pursuit + g-g-v 限速 profile（看地图/全状态/对手真值） |
| 学生 | 无状态帧堆叠 1D-CNN（~0.8M 参数），输入 = 多层 lidar 通道 + 速度 + opponent_state，输出 (δ, v) |
| 训练范式 | BC（MSE 监督）+ DAgger（滚学生→查专家→聚合→重训），**无 RL** |
| 仿真 | `~/f1tenth_gym`（f110_gym 0.2.1），系统 Python 3.8（gym0.19/torch2.1/numpy1.22/scipy1.10/numba0.56） |
| 地图 | 用户后续自放；先用 gym 自带 + 随机赛道 smoke-test |
| 验证深度 | 离线只做 smoke-test；GPU 训练 / TensorRT / 实车留给用户按指南执行 |

---

## 2. 关键诚实约束（必须写进论文）

1. **`f110_gym` 是 2D 仿真，无高度维度。** 「3D 多层感知」(创新点1) 的收益**无法在仿真内学习/度量**。
   - 学生网络做成**层数无关**：仿真里 lidar 输入 = 1 通道（gym 2D scan）；实车 = 3 通道（`/scan_3d` 的 low/body/high）。
   - §5 的「感知维度」消融因此是**实车 / 定性**结论（与文档自身「对低/高障碍区分正确率」一致）。
   - **可在仿真内做**的消融：BC vs DAgger、sysid 接地 vs gym 默认参数、g-g-v 护盾 on/off、反应式 vs BC 后端。
2. **sysid 横向 `ay_max` 仍是假设值**（W2 未补绕圆 bag）→ 接地仿真的横向极限标注 `ASSUMED`，绕圆 bag 补测后替换。
3. **BC 天花板 = 专家水平**；学生很难系统超过老师。超越专家留作可选「薄残差 RL polish」（不在本期）。

---

## 3. 包结构

```
src/slash_distill/
  package.xml  setup.py  setup.cfg  resource/slash_distill
  slash_distill/
    __init__.py
    params.py                  # 载入 sysid 接地参数（来自 config/sysid_params.yaml）+ NUMBA_CACHE_DIR 兜底
    common/
      __init__.py
      ggv.py                   # 纯py g-g-v 投影（移植 ggv_shield_node._shield，sim+部署共用）
      reactive.py              # 反应式控制律（FTG+居中+PD+曲率限速，doc§1.1）→ 专家候选 & 部署兜底
      obs.py                   # 学生观测构造：lidar 分层重采样到定长 + 速度 + opponent_state(W4 布局)
    sim/
      __init__.py
      grounded_env.py          # 包 f110_gym：注入 sysid 参数 + 命令侧 K/τ/Td；统一 reset/step/obs 接口
      opponent.py              # 仿真里对手车（第二 agent）用反应式/pursuit 驱动，产 opponent_state 真值
    experts/
      __init__.py
      pursuit_expert.py        # 特权专家：中心线 pure-pursuit + g-g-v 限速（看地图/全状态/对手真值）
      centerline.py            # 从地图 yaml/占据栅格抽中心线 waypoints（无中心线则用反应式专家兜底）
    models/
      __init__.py
      student_cnn.py           # 无状态 1D-CNN（LidarNet 思路），输入(通道lidar+标量)→(δ,v)
    # ---- 脚本（python -m slash_distill.xxx） ----
    collect_demos.py           # W5：多地图+域随机化跑专家采 (obs_student, action_expert*) → .npz
    train_bc.py                # W6：BC(MSE) 训练 + train/val + ckpt
    eval_closed_loop.py        # W6：grounded_env 闭环评估（完成率/碰撞/圈速）；护盾&兜底可开关
    dagger.py                  # W7：DAgger 迭代
    export_onnx.py             # W7：ckpt → ONNX（形状/动态轴）
    build_trt.sh               # W7：ONNX → TensorRT（trtexec）+ 时延基准
    run_ablations.py           # W8：§5 消融矩阵 → CSV
    make_figures.py            # W8：CSV → 图表/表格
    nodes/
      __init__.py
      student_policy_node.py   # W7 部署：/scan_3d+/odom+/opponent_state → ONNX/TRT → /drive_raw；超时/低置信→反应式兜底
  config/
    sysid_params.yaml          # 接地车辆参数（K/τ/Td、L、a_max、s±、ay、v_max…）
    distill.yaml               # 数据/训练/学生超参/观测维度
  launch/
    race_student.launch.py     # /scan_3d → student_policy_node → ggv_shield → mux（含兜底）
```

---

## 4. 接口契约（与 W1–W4 对齐，杜绝漂移）

### 4.1 学生观测 `obs.py`（部署=训练同构）
- `lidar`：把来源 scan（仿真 1080@270° / 实车 `/scan_3d` 360°@0.0043）**重采样到定长 `n_beams`（默认 108）× `n_layers` 通道**。仿真 `n_layers=1`；实车 `n_layers=3`（low/body/high）。
- `scalars`：`[ego_speed]` + `opponent_state[10]`（W4 布局：valid,range,bearing,x,y,vx,vy,speed,range_rate,is_dynamic）。可配 `use_opponent`。
- 归一化：lidar / `range_max`；速度 / `v_max`；opponent 量纲各自缩放。固定在 `obs.py` 一处。

### 4.2 动作
- 学生输出与专家标签都是 **(steering_rad, speed_mps)**；训练前各自按 `s_max=0.34 / v_max=2.0` 归一化到 [-1,1]。

### 4.3 g-g-v `common/ggv.py`
- 纯函数 `project(steer, v_req, v_ref, dt, limits) -> (steer, v, v_cap)`，数学**逐行对齐** `ggv_shield_node._shield`：
  - `|δ|≤steer_limit`；`κ=tan|δ|/L`；`v≤sqrt(ay_max·factor/κ)`；`dv∈[-ax_brake·dt, ax_accel·dt]`。
- 部署节点**不**自己投影（沿用现成 `slash_safety/ggv_shield`，学生发 `/drive_raw`）；仿真用本模块在 rollout 里投影。

### 4.4 接地 `sim/grounded_env.py`
- gym `params`：`lf+lr=0.25`（替换默认 0.33）、`s_min/s_max=±0.34`、`a_max=ax_accel`、`v_max`、`mu` 调到 `ay_max/g`。
- 命令侧：稳态增益 `K=0.9139`（指令速度×K）+ 一阶滞后 `τ` / 纯延迟 `Td`（命令缓冲），逼近 sysid 纵向模型。
- `default` 模式：保留 gym 原始默认（0.33m/未标定）→ 供「sysid vs 默认」消融对照。

---

## 5. 数据流

**训练采集（W5）**
```
grounded_env(map_i, 域随机化) → 专家(看全状态/中心线/对手真值) → action*
                              → obs.py(可部署观测) → 存 (obs, action*)
```
**部署（W7）**
```
/scan_3d(多层) + /odom + /perception/opponent_state
   → student_policy_node(obs.py → ONNX/TRT, 50Hz) → /drive_raw
   → ggv_shield(/drive_raw→/drive) → ackermann_mux(navigation)
学生超时/低置信 → 1 帧内切 common/reactive.py 兜底（off/steer_only/full 思想）
```

---

## 6. §5 消融矩阵（W8）落地

| 维度 | A | B | 指标 | 在哪做 |
|---|---|---|---|---|
| 车辆参数 | gym 默认(0.33m) | sysid 接地 | 圈速/完成率/轨迹一致性 | **仿真** |
| 安全壳 | 无 g-g-v | 有 g-g-v 投影 | 越界动作率/平滑度/圈速 | **仿真** |
| 控制后端 | 纯反应式 | base+BC 学生 | 圈速/完成率/时延/回退率 | **仿真** |
| 训练范式 | BC | DAgger | 完成率/协变量漂移 | **仿真** |
| 感知维度 | 单层 2D | 3D 多层 | 低/高障碍区分 | **实车/定性**（2D 仿真无法度量） |
| 学生结构 | （可选）GRU | 无状态 CNN | 时延/TRT 兼容 | 仿真+Jetson |

`run_ablations.py` 跑可仿真的 4 行 × N 地图 × 多 seed → CSV；`make_figures.py` 出表/图。

---

## 7. 验证（离线 smoke-test，本期交付）

- 全部 `*.py` 过 `py_compile` / import。
- `grounded_env` reset+step 数十步无异常；专家能驱动跑一小段。
- `collect_demos` 采 ~200 帧小数据集落盘。
- `student_cnn` 前向 + 1 步 BC 训练 loss 下降。
- `eval_closed_loop` 跑 1 个短 episode 出指标。
- `export_onnx` 导出且 onnx 形状校验（**onnx 未装 → 跳过并提示**）。
- `student_policy_node` import（rclpy 在则可起，缺 onnxruntime/TRT 时走 numpy 兜底前向或反应式）。
- `run_ablations` 以「极小 steps」跑 2 行出 CSV。

> 真正的多地图大数据采集、长训练、DAgger 多轮、trtexec、实车闭环 = 用户按各周《使用指南》执行。

---

## 8. 交付物清单

- 代码：`src/slash_distill/` 全套（§3）。
- 文档：`W5_接地仿真与特权专家_使用指南.md`、`W6_BC学生蒸馏_使用指南.md`、`W7_DAgger与部署集成_使用指南.md`、`W8_消融与写作_使用指南.md`、`周总结3.md`；并补完 `F1TENTH_部署时间安排.md` §9「下一步」+ W5–W8 完成状态表。
