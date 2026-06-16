# F1TENTH 创新点方案时间安排

> 本文档的目标：在 **2 个月** 内，给出一条 **可靠（低风险、结果可复现）** 的创新主线，
> 并把 `/studies` 下已有的全部代码与数据资产串成一条互相补强的链。
>
> 它是前两篇方案的**收敛版与落地版**，与它们的关系：
> - [F1TENTH_3D残差竞速_α-RPO方案.md](F1TENTH_3D残差竞速_α-RPO方案.md) —— α-RPO 衰减残差路线，本文把它从主线**降级为"可选加分"**。
> - [F1TENTH_部署优先特权蒸馏方案.md](F1TENTH_部署优先特权蒸馏方案.md) —— 本文采纳它的核心判断（BC/蒸馏比残差 RL 更好部署），并把它作为"加分层"的方法。
>
> 平台：Jetson Orin NX 16GB + Livox MID-360（三维）+ VESC，Ubuntu 20.04 / ROS2 Foxy。

---

## 0. 已拍板的三条决策（本文档据此成型）

| 维度 | 选定 | 含义 |
|---|---|---|
| **主线方法** | 强经典主干 + 学习加分 | 3D 反应式 + sysid 接地 + g-g-v 安全壳为**可靠主干**；BC/蒸馏作上层加分，失败也不影响 |
| **验证范围** | 仿真为主 + 实车单车低速演示 | 主结果在 f1tenth_gym（sysid 接地）里出；实车只做**单车、低速**跑通 / 避静态&单动态障碍演示 |
| **学习路线** | 模仿 / 蒸馏（BC + DAgger） | 不用 PPO，无 RL gap、确定性监督学习、最易复现 |

**这套配置的可靠性来自一个结构性原则**（见 §2）：**主干是已经能跑的经典反应式策略，学习只在它之上叠加、且永远有反应式兜底 + g-g-v 物理护盾**。即使学习部分效果一般，主体贡献与实验结论依然成立。

---

## 1. 现有资产盘点：你已经站在一个"半成品"上

可靠的前提是**别从零开始**。`/studies` 下已经有的、可直接复用的东西：

### 1.1 已能跑的无地图反应式策略（主干底座）

- [battle_fast2_node.py](f1tenth_car_ws/src/car_navigation/battle_fast2_node.py)（650 行，原始核心）与其产品化重构版
  [reactive_racing](f1tenth_car_ws/src/car_navigation/reactive_racing/)（1470 行，全参数化、分辨率无关、日志节流）。

  其控制律本质是 **Follow-the-Gap + 走廊居中 + PD + 曲率相关限速**，第一性原理拆解：

  - **方向项**（朝最优缝隙转）：
    $$\text{term1} = -\max\!\Big(e^{-d_{\max}/D_{th}},\,0.7\Big)\cdot\frac{i_{gap}-i_{center}}{360}\pi$$
    $(i_{gap}-i_{center})$ 是所选缝隙相对正前方的角向偏差；$e^{-d_{\max}/D_{th}}$ 在前方很开阔时**衰减转向**（空旷就别乱打），下限 0.7。
  - **居中项**（走廊对中）：
    $$\text{term2}=\frac{d_{left}-d_{right}}{d_{left}+d_{right}}$$
    左右距离归一化不对称度——等价于一个**斥力居中**项。
  - **PD 转向**：$\delta = P\cdot\text{angle} + D\cdot(\text{angle}-\text{angle}_{prev})$。
  - **限速律**：$v = s\,(w_d\,e^{-|\text{angle}|}+w_b)$，转角越大越慢。

  > 这正好是 RaceMOP 里 **APF 基础策略**的"手写版对位件"：缝隙方向≈引力、障碍膨胀≈斥力。所以它天然适合当
  > 蒸馏专家的对照基线、以及学习层的安全兜底。

### 1.2 已做了一半的系统辨识 + g-g-v（主干的"物理接地"）

[chassis_sysid.py](sysid/chassis_sysid.py) 已经从 `chassis_response_test` / `chassis_test` 两个 rosbag 里辨识出：

- **纵向一阶+延迟模型** $\dot v=(K\,u(t-T_d)-v)/\tau$ 的 $K,\tau,T_d$，并报告了拟合 $R^2$；
- **稳态增益 ≈ 0.93** → 标定 `speed_to_erpm_gain=4614` 基本成立（约 7% 跟踪损失）；
- **纵向加减速能力** $a_{x,\max}^{accel}, a_{x,\max}^{brake}$（实测，g-g-v 的纵轴）；
- **g-g-v 椭圆 + 限速 profile** $v_{\max}(\kappa)=\min(v_{cfg},\sqrt{a_{y,\max}/\kappa})$。

它还**诚实标注了两处缺口**（这恰好是你要补的工作量）：
1. 转向标定测试车速 ≤ 0.31 m/s（近静止），**转向→横摆增益、有效轴距不可辨识** → 需补一个**有速度的绕圆 bag**；
2. **横向摩擦极限 $a_{y,\max}$ 不可实测**，现按 $\mu g$ 参数化 → 需 **skidpad / 绕圆数据**标定。

### 1.3 三维雷达——但现在 3D 信息被白白丢掉了

真实管线（[pointcloud_to_laserscan_launch.py](f1tenth_car_ws/src/car_perception/pointcloud_to_laserscan/launch/pointcloud_to_laserscan_launch.py)）是
**直接吃 `/livox/lidar`、用一层 `min_height=-0.05 / max_height=0.10` 的薄高度切片**压成 2D `/scan`
（360°、≈0.0043 rad/点、`use_inf=True`）。

> ⚠️ 注意：本 checkout 里**没有 linefit 地面分割包**，去地面靠的就是这条 ±薄片。
> **这意味着 MID-360 的全部高度结构 / 360° / 非重复扫描时间特性都被扔了**——而这正是你相对全网 2D LiDAR 工作的**唯一硬件差异化**。

### 1.4 两个纯仿真训练仓库（学习加分层的素材）

- [raceMOP-main](../raceMOP-main/)：APF 基础策略 + 残差 PPO + 截断高斯（纯 f1tenth_gym 仿真，PyTorch/JAX）。
- [end2/End2Race](../end2/End2Race/)：GRU 端到端模仿学习超车（BC，仿真）+ lattice/pure-pursuit 专家。

  二者都**不是 ROS2 包、不能直接上车**，属于"离线训练轨"。本文只取其中**部署友好**的部分（见 §3.4）。

### 1.5 当前代码里两个"已知会坏"的点（= 现成的改进入口）

1. `battle_fast2` 的 `DynamicObastcle()` 用 **intensity 均值差 > 5** 判动态障碍——MID-360 反射率(0–255)与 2D 雷达 intensity **不可比**，**上实车不可用**（reactive_racing 已部分改成运动学跟踪，但还可做成"单帧测速"的真创新）。
2. 文献（RaceMOP/End2Race）里**硬编码车辆参数**（轴距 0.33 m、横向 0.8g）——你的车轴距是 **0.25 m**，极限要用 §1.2 的实测值。

---

## 2. 一个决定可靠性的核心判断

> **可靠 = 一个"已经能跑"的保底主体 + 一个"做成了就加分、没做成也不致命"的上层。**

把它落到这台车上：

1. **主体是无地图反应式策略**（§1.1，已能跑）。它不依赖地图、不依赖网络、不依赖训练，是**永远可用的安全兜底**。
2. **学习层（BC 学生网络）只在主体之上叠加**：学生不可信 / 不确定 / 推理超时 → 1 帧内回退反应式（这套"可切换后端 + 回退"骨架你在 [mpc_execplan.md](f1tenth_car_ws/mpc_execplan.md) 里已经设计过：`off / steer_only / full` + `mpc_ok`）。
3. **g-g-v 物理护盾在最后兜底**：无论上层输出什么，都投影回**实测可行域**（§1.2）。这是一层与学习正交的、可解释的安全保证。

这样，**失败模式被隔离了**：学习失败 → 退化成"3D 反应式 + sysid + 安全壳"，依然是一篇完整、有实测、有消融的工程论文；学习成功 → 锦上添花、新颖性拉满。**这就是"可靠创新点"的工程含义。**

---

## 3. 创新点（分层，按"必成→加分"排列）

四个创新点，前三个是**必成主干**（基于已有代码扩展，2 个月内可控），第四个是**可降级加分**。

### 创新点 1（主干·必成）：三维多层感知的无地图反应式竞速（3D-MLR）

**做什么**：把现在"单层薄切片 → 单条 2D `/scan`"升级为**多高度层带**（如地面附近 / 车体高度 / 车顶高度若干 band），
让反应式前端**同时**看到多条 scan，从而区分**低路肩 / 车高障碍 / 悬空物**，并对"真正会挡路"的障碍做更准的缝隙选择与膨胀。

**为什么可靠**：它是对 [reactive_racing](f1tenth_car_ws/src/car_navigation/reactive_racing/) 前端的**增量扩展**，不是新算法；
多开几个 `pointcloud_to_laserscan` 实例（不同 `min/max_height`）或写一个轻量多层投影节点即可，**不碰 RL**。

**新颖性（诚实）**：全网 F1TENTH 竞速几乎全是 2D 平面 LiDAR；**"用 MID-360 多层结构做反应式竞速决策"在该平台上基本空白**。
这是平台差异化，不是算法噱头。

**复用资产**：`/livox/lidar`、pointcloud_to_laserscan、reactive_racing 前端 `frontend_scan_process`。

---

### 创新点 2（主干·必成）：Livox 单帧测速替换 intensity 动态判据

**做什么**：删掉不可移植的 intensity 动态判据，改用 **MID-360 非重复扫描的逐点时间戳 + 多帧运动学跟踪**
估计前方障碍的**接近速度 / 横向速度**（VeloVox 思路的轻量化版），作为动态障碍 / 是否减速跟随的依据。

**为什么可靠**：reactive_racing 里已有多帧数据关联 + age/missed_frames 的跟踪雏形
（`_update_dynamic_obstacle_tracks`），把它的判据从 intensity 换成**纯运动学（index_rate / range_rate / 单帧测速）**即可；
**单车场景**就能验证（用一个被推动的纸箱 / 移动障碍即可，不需要第二台竞速车）。

**新颖性（诚实）**：单帧/短时测速本身不新，但**在 3D MID-360 上、用它替换反应式竞速里的 intensity 判据**，是修一个真实 bug + 平台特性利用，边界清晰、可量化。

**复用资产**：reactive_racing 的跟踪器、MID-360 逐点时间戳。

---

### 创新点 3（主干·必成）：sysid 接地 + 实测 g-g-v 安全壳

**做什么**：
1. **补全 sysid**：采一个**有速度的绕圆 / skidpad bag**，补 [chassis_sysid.py](sysid/chassis_sysid.py) 自己标注缺的两项（转向→横摆增益、横向 $a_{y,\max}$）。
2. **落地 g-g-v 安全壳**：把规划/学习输出的 $(\delta, v_{cmd})$ 投影回**实测可行域**：
   $$\kappa=\frac{\tan\delta}{L},\quad L=0.25\text{ m};\qquad v_{safe}\le\sqrt{\frac{a_{y,\max}}{\max(\kappa,\epsilon)}};\qquad \Delta v\in[a_{x}^{brake},\,a_{x}^{accel}]\cdot \Delta t$$
   （这正是 [F1TENTH_部署优先特权蒸馏方案.md](F1TENTH_部署优先特权蒸馏方案.md) §3 给出的替换代码，本文把它作为正式安全模块落地。）
3. **接地仿真**：把实测 $K,\tau,T_d$ 与 g-g-v 灌进 f1tenth_gym，替换文献的硬编码 0.33 m / 0.8g。

**为什么可靠**：§1.2 已经做了 60%，剩下是"采一个 bag + 写一个投影滤波器 + 改仿真参数"，**全是确定性工程**，不会"训不出来"。

**新颖性（诚实）**：在线 1 分钟 sysid（ForzaETH）是邻近工作；你的差异是**离线 sysid → 同时接地仿真与安全壳**，
并**实证文献硬编码参数与本车的失配**（0.33 vs 0.25 m，假设 0.8g vs 实测）。这是 sim-to-real 章节的硬证据。

**复用资产**：chassis_sysid.py、chassis_test / chassis_response_test bag、VESC 真实标定。

---

### 创新点 4（加分·可降级）：特权专家 → 学生 BC/DAgger 蒸馏（3D 多层输入 + 反应式兜底 + g-g-v 护盾）

**做什么**：在 **sysid 接地的 f1tenth_gym** 里，用一个**特权专家**（可用地图 / 赛车线 / 完整车体状态 / 障碍真值——部署时拿不到的信息）产生示范，
再用 **BC + DAgger** 把它蒸馏成一个**轻量无状态学生网络**：

- **学生观测**（全可测）：3D 多层 scan（创新点 1）+ 自车速度（+ 创新点 2 的对手运动通道，可选）；
- **学生输出**：$(\delta, v)$；
- **部署形态**：ONNX→（Jetson 本机）TensorRT，50 Hz 推理；**学生不可信/超时 → 退反应式**；**输出过 g-g-v 护盾**。

第一性原理：
- **BC 目标**：$\min_\theta\,\mathbb E_{(s,a^\*)\sim\mathcal D_{exp}}\|\pi_\theta(s)-a^\*\|^2$（确定性监督，无奖励、无 RL gap）。
- **协变量漂移**：学生会走到专家没见过的状态，误差按 $O(\varepsilon T^2)$ 累积。
- **DAgger 修正**：滚动学生、在其访问状态上**重新查询专家**、聚合再训，误差降到 $O(\varepsilon T)$——且仍不需要奖励函数。
- **特权非对称**：专家看特权状态、学生只看可部署观测——这就是 [α-RPO 方案](F1TENTH_3D残差竞速_α-RPO方案.md)"训练特权、部署丢弃"的思想，但用**模仿**而非 PPO 实现，更易复现。

**为什么可靠**：选 BC/DAgger 而非残差 RL，正是 [部署优先特权蒸馏方案](F1TENTH_部署优先特权蒸馏方案.md) §1 的结论——
**无 PPO、无奖励整形、无 RL sim-to-real gap、部署只跑一个网络**。且**有反应式兜底**：学生效果差也不致命。

**新颖性（诚实）**：BC 超车 End2Race 做过、特权蒸馏也非全新。**你的组合点**：
`3D 多层 LiDAR 学生输入 × sysid 接地的特权专家 × g-g-v 护盾 × 反应式安全兜底 × 单车`——这个组合是空的。
**但务必把它定位成"加分"**，主贡献仍在创新点 1–3 + §5 的消融。

**复用资产**：End2Race 的专家/BC 框架、RaceMOP 的无状态 1D-CNN 思路（比 GRU 更 TensorRT 友好）、mpc_execplan 的可切换后端骨架。

---

## 4. 诚实的新颖性校准（评审会打的点，先自己想清楚）

| 说法 | 成不成立 | 修正表述 |
|---|---|---|
| "首个用 3D LiDAR 做竞速决策" | ❌ 已有 3D-LiDAR + DDPG 端到端上真车 | "**首个 3D 多层 LiDAR 反应式 / 蒸馏**竞速（非端到端 RL）" |
| "残差 RL 上实车很新" | ❌ RLPP / On-Board SAC / α-RPO 已做 | 本文**不走残差 RL 主线**，规避此拥挤赛道 |
| "BC 超车很新" | ❌ End2Race 做过 | 新在**组合**：3D 输入 + sysid 专家 + 护盾 + 兜底 + 单车 |
| "在线 sysid 很新" | ❌ ForzaETH | 新在**离线 sysid 同时接地仿真与安全壳 + 实证文献参数失配** |

> **真正的护城河 = 组合 + 平台（3D MID-360）+ 物理接地 + 系统消融**，不是任何单点的"首个"。
> 这种定位**对硕士是足够且稳妥的**——评审看重的是"做扎实、讲清楚、有实测、有对照"，而非一个高风险的"首创"。

---

## 5. 实验与消融设计（可靠性的真正后盾）

**这一节是关键**：即使创新点 4 的学习效果平平，**消融实验本身就是可发表的结论**——"在一台 3D-LiDAR F1TENTH 上，到底什么对竞速性能与 sim-to-real 有用"。这保证了"无论如何都有结果"。

**主消融表**（仿真为主，跑在 sysid 接地的 f1tenth_gym；实车做单车低速复核）：

| 消融维度 | 对照 A | 对照 B | 量化指标 |
|---|---|---|---|
| 感知维度 | 单层 2D `/scan` | **3D 多层（创新点 1）** | 完成率、碰撞率、对低/高障碍的区分正确率 |
| 动态判据 | intensity（原） | **运动学/单帧测速（创新点 2）** | 动态障碍误报/漏报率、跟随触发正确率 |
| 车辆参数 | 文献硬编码(0.33m/0.8g) | **sysid 实测(创新点 3)** | sim-to-real 圈速差、轨迹一致性 |
| 安全壳 | 无 | **g-g-v 投影(创新点 3)** | 越界动作率、平滑度、圈速折中 |
| 控制后端 | 纯反应式（base） | **base + BC 学生（创新点 4）** | 圈速、完成率、推理时延、回退触发率 |

**核心指标**：单圈时间、完成率/碰撞率、sim-to-real 差距（同策略 sim vs 实车低速）、推理时延（Jetson 上 <20 ms / 50 Hz）、安全壳触发统计。

**验证平台**：
- **仿真（主）**：f1tenth_gym + sysid 接地参数，多张赛道、域随机化。
- **实车（演示）**：单车、低速；静态障碍避让 + 单个移动障碍跟随/避让；先 ~1 m/s 跑通再逐步提速（按 skill 安全清单 + 遥控接管/estop）。

---

## 6. 两个月周排期（8 周，每周都有降级点）

> 原则：**先把"必成"做完拿到结果，再做"加分"**；每周产出独立成立，某周滑坡不影响已交付部分。

| 周 | 目标 | 产出 | 降级点（滑坡时的兜底） |
|---|---|---|---|
| **W1** | 实车反应式跑通 + 部署底座 | 整合 launch（雷达→p2l→bringup→reactive_racing）+ 保守速度参数 + 实车单车低速跑通 | 已基本可跑，几乎零风险 |
| **W2** | sysid 补全 + g-g-v 安全壳 | 采绕圆 bag，补横向 $a_{y,\max}$/转向增益；g-g-v 投影滤波器落成模块 | 即使横向标定不理想，纵向+假设 $\mu$ 仍可用（已实现） |
| **W3** | 3D 多层感知节点（创新点 1） | 多高度层投影节点 + 接进反应式前端 + RViz 验证 | 退回"双层"甚至沿用单层，主干仍成立 |
| **W4** | Livox 测速（创新点 2）+ 实车演示 | 运动学/单帧测速替换 intensity；实车避静态+单动态障碍演示 | 退回纯多帧跟踪（已有雏形） |
| **W5** | 接地仿真 + 特权专家 | f1tenth_gym 灌 sysid 参数；搭特权专家（End2Race lattice / 赛车线）采示范 | 专家可先用现成 pure-pursuit/反应式，降复杂度 |
| **W6** | BC 学生训练（创新点 4） | 无状态 1D-CNN 学生（3D 多层输入）BC 训练 + 仿真闭环评估 | 学生若不收敛，加分层标记为"未达标"，主干照常 |
| **W7** | DAgger + 部署集成 | DAgger 迭代；ONNX→TensorRT；学生+反应式兜底+g-g-v 护盾集成；实车单车低速复核 | 实车若不稳，主结果用仿真，实车仅演示 base |
| **W8** | 消融全跑 + 写作 | §5 消融表全部跑出 + 数据/图表整理 + 论文初稿 | — |

**关键里程碑**：W4 末"必成主干"全部完成（创新点 1–3 + 实车演示）→ **此时已"安全"**；W5–W7 是加分；W8 收口。

---

## 7. 资产 → 创新点映射（让方案"便宜"）

| 资产 | 位置 | 喂给哪个创新点 |
|---|---|---|
| 反应式策略（已能跑） | [battle_fast2_node.py](f1tenth_car_ws/src/car_navigation/battle_fast2_node.py) / [reactive_racing](f1tenth_car_ws/src/car_navigation/reactive_racing/) | 主干底座 + 学习兜底 + 蒸馏对照基线 |
| 多帧跟踪雏形 | reactive_racing `_update_dynamic_obstacle_tracks` | 创新点 2 |
| 系统辨识 + g-g-v | [chassis_sysid.py](sysid/chassis_sysid.py) + chassis bags | 创新点 3 |
| g-g-v 替换代码 | [部署优先特权蒸馏方案.md](F1TENTH_部署优先特权蒸馏方案.md) §3 | 创新点 3 安全壳 |
| 3D 点云入口 | `/livox/lidar` + [pointcloud_to_laserscan](f1tenth_car_ws/src/car_perception/pointcloud_to_laserscan/) | 创新点 1 |
| 可切换后端 + 回退骨架 | [mpc_execplan.md](f1tenth_car_ws/mpc_execplan.md)（`off/steer_only/full` + `mpc_ok`） | 创新点 4 部署集成 |
| BC 专家/框架 | [End2Race](../end2/End2Race/)（lattice/pure-pursuit + BC） | 创新点 4 专家与蒸馏 |
| 无状态 1D-CNN 思路 | [raceMOP](../raceMOP-main/) `nn.py` LidarNet | 创新点 4 学生骨干（比 GRU 更 TRT 友好） |
| 驱动栈（已配好） | livox / vesc / ackermann_mux / p2l | 全程实车底座 |

---

## 8. 风险登记与降级策略

| 风险 | 影响 | 降级 / 规避 |
|---|---|---|
| 学生网络训不好 / 蒸馏不收敛 | 创新点 4 失败 | **主干已成立**；加分层标"未达标"，消融里作为负结果照样写 |
| 实车提速后不稳 | 实车结果弱 | 主结果放仿真；实车只承诺**单车低速演示**（已选定范围） |
| 横向 $a_{y,\max}$ 标不准 | g-g-v 偏保守 | 沿用 $\mu g$ 参数化（已实现），并在论文里诚实标注 |
| 3D 多层增加点云开销 | 50 Hz 预算紧 | 降层数 / 降频；Jetson 先 `nvpmodel -m 0 && jetson_clocks` 锁频 |
| 自车体在 360° scan 里成假墙 | 反应式误判 | 按角度+距离画自车 box 丢点；抬 `range_min` |
| Foxy 依赖 / 编译坑 | 拖进度 | 驱动已 vendored；`colcon build --symlink-install`，rosdep 超时改手动 apt |

---

## 9. 一句话总结 + 下一步

**一句话**：以"3D 多层感知 + Livox 测速 + sysid 接地 g-g-v 安全壳"的**可靠经典主干**为主体（创新点 1–3，基于已能跑的代码扩展、2 个月内必成），
在其上叠加"特权专家 → BC/DAgger 学生 + 反应式兜底 + 护盾"的**可降级加分层**（创新点 4）；
主干失败概率极低、加分层失败也不致命，因此**无论如何都有一篇有实测、有对照、有消融的完整工程论文**。

**下一步（W5–W8 已落地，按周推进收尾）**：

1. 放入自有赛道 + 中心线 CSV → 采专家示范（W5，`collect_demos`）。
2. GPU 上正式 BC 训练 + 闭环评估（W6，`train_bc` / `eval_closed_loop`）。
3. DAgger 多轮 + ONNX→TensorRT + 实车单车低速复核（W7，`dagger` / `export_onnx` / `build_trt.sh` / `race_student.launch.py`）。
4. 仿真 4 维消融全跑 + 实车 3D/sim-to-real 素材 + 论文初稿（W8，`run_ablations` / `make_figures`）。

---

## 10. W5–W8 落地状态（2026-06-15 更新）

W5–W8 软件全部交付在单一包 `src/slash_distill`（离线训练 + ROS2 部署同包），离线 smoke-test 全通过。
详见 [设计 spec](F1TENTH_W5-W8_蒸馏部署_设计spec.md)、各周使用指南与 [周总结3](周总结3.md)。

| 周 | 目标 | 交付物 | 状态 |
|---|---|---|---|
| **W5** | 接地仿真 + 特权专家 | `sim/grounded_env.py`、`experts/pursuit_expert.py`、`common/{ggv,reactive,obs}.py`、`collect_demos.py` + 配置 | 软件完成（已 smoke） |
| **W6** | BC 学生训练 | `models/student_cnn.py`、`train_bc.py`、`eval_closed_loop.py` | 软件完成（已 smoke） |
| **W7** | DAgger + 部署集成 | `dagger.py`、`export_onnx.py`、`build_trt.sh`、`nodes/student_policy_node.py`、`launch/race_student.launch.py` | 软件完成（已 smoke / Foxy import 通过） |
| **W8** | 消融全跑 + 写作 | `run_ablations.py`、`make_figures.py` | 软件完成（已 smoke） |

> 关键诚实点（写进论文）：f110_gym 为 **2D**，「3D 多层感知」消融为**实车/定性**；横向 `ay_max` 仍为假设值，
> 绕圆 bag 实测后替换。其余消融（车辆参数 / 安全壳 / 控制后端 / BC vs DAgger）在仿真内完成。
>
> 使用指南：[W5](W5_接地仿真与特权专家_使用指南.md) · [W6](W6_BC学生蒸馏_使用指南.md) · [W7](W7_DAgger与部署集成_使用指南.md) · [W8](W8_消融与写作_使用指南.md)