# W6 使用指南：BC 学生网络训练 + 闭环评估（创新点4）

> 工作区：`~/slash_ws/src/slash_distill`。承接 W5 的 `data/demos.npz`。
> 目标：把特权专家蒸馏成一个**无状态 1D-CNN 学生**（BC，确定性监督，无 RL gap），
> 并在 sysid 接地仿真里做闭环评估（带 g-g-v 护盾 + 反应式兜底）。

---

## 0. W6 要解决什么

BC 目标（确定性监督，无奖励、无 PPO）：

$$\min_\theta\ \mathbb{E}_{(s,a^*)\sim\mathcal{D}_{exp}}\ \lVert \pi_\theta(s) - a^* \rVert^2$$

- **学生输入**（全可测）：多层 lidar（仿真 1 通道 / 实车 3 通道）+ ego 速度 + opponent_state。
- **学生输出**：`(steering, speed)`（网络出 `[-1,1]`，反归一化到 `±0.34 rad / [0,2] m/s`）。
- **学生骨干**：帧堆叠 1D-CNN（`models/student_cnn.py`，RaceMOP LidarNet 思路）——无状态、无控制流分支，
  比 GRU 更易导出 ONNX/TensorRT，前向约 1–2 ms。

---

## 1. 训练

```bash
cd ~/slash_ws/src/slash_distill

# 正式训练（100 epoch，GPU 自动用，无 GPU 回退 CPU）
python3 -m slash_distill.train_bc --data data/demos.npz --epochs 100 --out ckpt

# smoke-test（已验证）
python3 -m slash_distill.train_bc --data /tmp/demo_smoke.npz --epochs 3 --batch-size 32 --out /tmp/ckpt_smoke --device cpu
```

输出 `ckpt/student_best.pt`：`state_dict + model_cfg + meta`（含 `n_layers/n_beams/scalar_dim/steer_max/v_max`），
后续 `eval / export_onnx / dagger` 都从中重建网络与反归一化。

> Jetson 上 `--device cuda`（默认）可走 GPU；若报 CUDA 初始化错误（如沙箱/无显卡），脚本自动回退 CPU。

---

## 2. 闭环评估（在接地仿真里跑学生）

链路：`obs → 学生 → [低置信/异常→反应式兜底] → [g-g-v 护盾] → env.step`

```bash
# 评估学生（默认 10 episode，带护盾+兜底）
python3 -m slash_distill.eval_closed_loop --ckpt ckpt/student_best.pt --episodes 10

# 纯反应式基线（消融 A 对照）
python3 -m slash_distill.eval_closed_loop --reactive --episodes 10

# 关护盾 / 关兜底（看裸学生）
python3 -m slash_distill.eval_closed_loop --ckpt ckpt/student_best.pt --no-shield --no-fallback

# smoke-test（已验证）
python3 -m slash_distill.eval_closed_loop --reactive --episodes 1 --max-steps 80
```

输出指标：`collision_rate / completion_rate / mean_lap_count / mean_speed /
mean_inference_ms / fallback_rate / shield_clip_rate`。

---

## 3. 学生网络调参（`config/distill.yaml` 的 `model` / `train`）

| 参数 | 作用 | 建议 |
| ---- | ---- | ---- |
| `model.conv_channels / kernel_sizes / strides` | CNN 主干 | 默认 [32,64,64]/[7,5,3]/[3,2,2] |
| `model.mlp_hidden` | 头部 MLP | [256,128] |
| `train.epochs / batch_size / lr` | 训练 | 100 / 256 / 1e-3 |
| `train.val_frac` | 验证集比例 | 0.1 |
| `obs.n_beams` | 须与 W5 数据一致 | 改了要重采数据 |
| `obs.frame_stack` | 历史帧堆叠（仍无状态） | 1；想要时序信息可 2–3 |

> 参数量随 `n_beams` / 通道数变化（默认配置 ≈ 0.2M）。想更接近 RaceMOP 的 ~0.8M，
> 加大 `conv_channels` / `mlp_hidden` 即可；TensorRT 友好性不变。

---

## 4. W6 完成判据（self-check）

- [x] `student_cnn` 前向 + BC 训练 loss 下降（已验证：train 0.24→0.06）。
- [x] `eval_closed_loop` 学生/反应式均能跑出指标（已验证）。
- [ ] 正式训练 100 epoch，val_mse 收敛。
- [ ] 闭环评估学生**完成率 ≥ 反应式基线**、`mean_inference_ms` 合理。
- [ ] 记录「裸学生 vs 学生+护盾+兜底」差异（为 W8 消融与论文）。

---

## 5. 诚实状态 & 与后续衔接

- **BC 天花板 = 专家水平**：学生通常不超过老师；要超越留作可选「薄残差 RL polish」（不在本期）。
- **协变量漂移**：纯 BC 会在专家没见过的状态累积误差（$O(\varepsilon T^2)$）→ **W7 用 DAgger 修正**到 $O(\varepsilon T)$。
- **学生 ckpt 直接喂 W7**：`export_onnx` 导出部署、`dagger` 迭代都用它。
