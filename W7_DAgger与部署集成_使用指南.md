# W7 使用指南：DAgger 迭代 + ONNX/TensorRT 部署集成（创新点4 落地）

> 工作区：`~/slash_ws/src/slash_distill`。承接 W6 的 `ckpt/student_best.pt`。
> 目标：用 DAgger 修正 BC 协变量漂移；把学生导出 ONNX→TensorRT；
> 上 `student_policy_node` 部署节点（学生 + 反应式兜底），经现成 `ggv_shield` 进 mux。

---

## 0. W7 要解决什么

1. **DAgger**（`dagger.py`）：滚学生 → 在其访问状态上**查询专家** → 聚合 → 重训。
   误差从 BC 的 $O(\varepsilon T^2)$ 降到 $O(\varepsilon T)$，**仍不需奖励函数、仍无 RL**。
2. **部署导出**：`export_onnx.py`（ckpt→ONNX）+ `build_trt.sh`（ONNX→TensorRT engine / 时延基准）。
3. **部署节点**（`nodes/student_policy_node.py`）：实车 50 Hz 跑学生，**超时/低置信/后端缺失→反应式兜底**，
   输出 `/drive_raw`，由 W2 的 `ggv_shield` 统一做物理护盾。

---

## 1. DAgger 迭代

```bash
cd ~/slash_ws/src/slash_distill

# 用 W6 的 BC ckpt 作起点，迭代 5 轮
python3 -m slash_distill.dagger --init-data data/demos.npz --init-ckpt ckpt/student_best.pt --out ckpt_dagger

# 没有 BC ckpt 也行（会先训一个）
python3 -m slash_distill.dagger --init-data data/demos.npz --out ckpt_dagger

# smoke-test（已验证）
python3 -m slash_distill.dagger --init-data /tmp/demo_smoke.npz --iters 1 --rollouts 2 --max-steps 50 --epochs 2 --out /tmp/ckpt_dagger --device cpu
```

每轮：第 i 轮专家接管概率 `beta = beta_decay**i`（学生逐渐主导执行，专家始终提供标签），
聚合数据写 `ckpt_dagger/aggregated.npz`，重训出 `ckpt_dagger/student_best.pt`。参数见 `config/distill.yaml` 的 `dagger`。

---

## 2. 导出 ONNX → TensorRT

```bash
# 1) ckpt -> ONNX（+ onnx.checker + onnxruntime 数值自检）
python3 -m slash_distill.export_onnx --ckpt ckpt_dagger/student_best.pt --out ckpt/student.onnx

# 2) ONNX -> TensorRT engine（Jetson 本机 trtexec，FP16）+ 时延基准
bash slash_distill/build_trt.sh ckpt/student.onnx ckpt/student.engine
```

依赖（Jetson 上一次性装好）：

```bash
pip3 install --user onnx                 # 已装 1.17（export + checker 用）
# onnxruntime：Jetson 建议用 NVIDIA 预编译 onnxruntime-gpu（jetson-zoo），可挂 CUDA/TensorRT EP
# trtexec：JetPack 自带，一般在 /usr/src/tensorrt/bin/trtexec
```

> 学生网络很小，Orin 上 FP16 时延应 **< 5 ms**（目标 < 20 ms / 50 Hz）。
> `export_onnx` 已验证：导出 `lidar[1,1,108]+scalar[1,11]→action[1,2]`，`onnx.checker 通过`。

---

## 3. 上车部署

前置：W1 底座在跑（雷达/p2l/odom/mux），且**不起反应式规划器**（由学生接管）：

```bash
# 终端1：W1 底座（不起 planner）
ros2 launch roboracer_china_2025 race_reactive_bringup.launch.py run_planner:=false

# 终端2：W3 多层 + W4 测速 + 学生 + W2 护盾（一条命令）
ros2 launch slash_distill race_student.launch.py model_path:=/abs/ckpt/student.onnx backend:=onnx
```

链路：

```
/livox/lidar → multilayer_scan → /scan_3d
   ├─ obstacle_tracker → /perception/opponent_state
   └─ student_policy_node(/scan_3d + /odom + opponent_state → ONNX/TRT) → /drive_raw
         → ggv_shield → /drive → ackermann_mux(navigation)
```

**安全降级（与 mpc_execplan 的 off/steer_only/full 思想一致）**：

| 情形 | 行为 |
| ---- | ---- |
| `model_path` 为空 / 后端载入失败 | 节点始终走**反应式兜底**（仍能开车） |
| 学生输出 NaN / 与反应式速度差 > `fallback_speed_jump` | 该帧切反应式 |
| 传感器超时（`sensor_timeout_s`） | 不发指令，下游 `ggv_shield` 看门狗刹停 |
| 任意输出 | 都过 `ggv_shield` 物理护盾再进 mux |

**多层学生**：默认 `scan_topics:=['/scan_3d']`（1 通道，匹配本期仿真学生）。
训练出 3 通道学生后，设 `scan_topics:=['/perception/scan_layer_low','/perception/scan_layer_body','/perception/scan_layer_high']`。

---

## 4. 部署节点关键参数

| 参数 | 默认 | 说明 |
| ---- | ---- | ---- |
| `model_path` | "" | `.onnx` / `.pt`；空=纯反应式 |
| `backend` | onnx | `onnx`(onnxruntime, 可挂 TensorRT EP) / `trt` / `torch` |
| `scan_topics` | [/scan_3d] | 1 层=融合；3 层=低/体/高 |
| `control_rate_hz` | 50 | 决策频率 |
| `sensor_timeout_s` | 0.3 | 传感器超时→停发（下游刹停） |
| `use_fallback` | true | 低置信/异常→反应式 |
| `fallback_speed_jump` | 1.5 | 学生 vs 反应式速度差阈值 |

---

## 5. W7 完成判据（self-check）

- [x] `dagger` 能迭代（已验证：BC→采学生 rollout→聚合→重训）。
- [x] `export_onnx` 导出 + checker 通过（已验证）。
- [x] `student_policy_node` 在 Foxy 下 import 通过（已验证）。
- [ ] `build_trt.sh` 在 Jetson 出 engine 且时延 < 20 ms。
- [ ] `race_student.launch.py` 架空/原地跑通：`/scan_3d`、`opponent_state`、`/drive_raw`、`/drive` 都有输出。
- [ ] 实车单车低速：`model_path:=""` 先验证反应式兜底链路；再挂学生 onnx 复核（手柄随时接管）。

---

## 6. 诚实状态 & 与后续衔接

- **部署只跑一个网络**：实车不在线跑 APF/JAX/lattice；学生前向 + g-g-v 投影即可。
- **onnxruntime/TRT 缺失不致命**：节点退化为反应式仍能开车（安全优先）。
- **结果喂 W8**：BC ckpt 与 DAgger ckpt 一起进 `run_ablations`（BC vs DAgger 一行）。
