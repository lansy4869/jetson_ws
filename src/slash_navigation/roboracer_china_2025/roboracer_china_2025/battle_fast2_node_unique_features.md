# battle_fast2_node.py 独有功能说明

本文只讲 `battle_fast2_node.py` 相比 `battle_fast2_node（复件）.py` 多出来的内容。  
复件保留了基础扫描、启发式转向、赛道线与简化 MPC，但原文件额外加入了“行为层 + 风险层 + 诊断层”。

## 1. 行为状态机

原文件定义了完整的行为状态：

- `CRUISE`
- `FOLLOW`
- `OVERTAKE_PREPARE`
- `OVERTAKE_EXECUTE`
- `RETURN_TO_RACELINE`

对应实现主要在 `_update_behavior_state()`，并由 `behavior_state`、`behavior_state_stamp`、`behavior_overtake_side`、`behavior_state_switch_count` 维护。

它的作用不是单纯“看见障碍物就绕开”，而是把控制流程拆成几段：

1. `CRUISE`：正常巡航。
2. `FOLLOW`：前方净空不足，但还没有适合超车的窗口，先降速跟随。
3. `OVERTAKE_PREPARE`：判断到一侧有更好的开口，开始准备超车。
4. `OVERTAKE_EXECUTE`：执行超车。
5. `RETURN_TO_RACELINE`：超车后回到主线。

这个状态机还会同步设置两个简化标志：

- `Follow`
- `chaoche`

它们直接影响后面的速度上限和控制打印。

## 2. 时序障碍物建模

原文件不是只看当前一帧激光，而是维护了一条障碍物历史：

- `obs_track_history`
- `last_obs_summary`

相关函数是：

- `_extract_obstacle_summary()`
- `_update_obstacle_track()`

它会从当前障碍物段里提取：

- 是否检测到障碍物
- 障碍物中心角度
- 距离
- 宽度
- 相对速度 `rel_speed`
- 跟踪帧数 `track_count`
- 是否动态 `is_dynamic`

动态障碍物判定用的是“连续帧距离变化”和“中心角变化”两类信息，不是单帧阈值。

相关参数：

- `temporal_obs_enable`
- `obs_history_size`
- `obs_rel_speed_follow_threshold`
- `obs_min_track_frames`

## 3. 风险感知 MPC

这是原文件最核心的增强之一。

相关参数：

- `risk_enable`
- `risk_safe_distance`
- `risk_weight`
- `risk_collision_penalty`

相关函数：

- `_query_obstacle_distance()`
- `rollout_cost()`
- `solve_mpc()`

原文件在 MPC rollout 里不只算轨迹误差，还会算“预测轨迹点到障碍物的安全余量”：

- 余量 < 0：直接加大碰撞惩罚
- 余量 > 0 但小于安全距离：按距离做软惩罚

`solve_mpc()` 也不只是返回“能不能解出来”，还会返回 `min_margin`，也就是这次候选控制里最小的风险余量。  
这让节点可以在日志和话题里看到“这次控制到底有多贴边”。

## 4. 行为感知的轨迹和速度调制

原文件在生成局部参考轨迹时，会根据行为状态做额外偏置：

- `_apply_behavior_heading_bias()`

当状态处于超车相关阶段时，会按 `overtake_side_bias_deg` 给目标航向加一个左右偏置，避免轨迹太“死板”。

同时，控制阶段会按行为状态动态压速度：

- `FOLLOW` 时使用 `follow_speed_cap`
- `OVERTAKE_PREPARE / OVERTAKE_EXECUTE` 时使用 `chaoche_speed_cap`

这部分让速度控制不再只依赖曲率和 MPC 结果，而是受行为状态约束。

## 5. 额外诊断话题

原文件发布的诊断信息更多，主要是为了在线观察行为层和风险层状态。

新增/强化的话题包括：

- `battle_fast2/behavior_state`
- `battle_fast2/behavior_state_switch_count`
- `battle_fast2/obs_distance_m`
- `battle_fast2/obs_rel_speed_mps`
- `battle_fast2/obs_is_dynamic`
- `battle_fast2/front_clearance_m`
- `battle_fast2/risk_min_margin_m`
- `battle_fast2/steering_cmd_rad`
- `battle_fast2/speed_cmd_mps`

此外，`diagnostic_timer_callback()` 的日志也更完整，会打印：

- 当前速度
- MPC 求解时间
- 控制模式
- 行为状态
- 前向净空
- 风险余量

## 6. 前向净空与超车开口评估

原文件增加了两个用于行为决策的辅助量：

- `_front_clearance()`
- `_best_opening_side()`

它们分别负责：

- 计算车头正前方最小净空
- 比较左右两侧开口大小，选择更适合超车的一侧

这两个量是 `_update_behavior_state()` 的关键输入，决定什么时候从 `CRUISE` 转入 `FOLLOW`，以及超车往哪边走。

## 7. 可调的障碍物方差过滤

原文件保留了 `obs_variance_threshold` 这个参数，并在 `filter_obstacles_by_variance()` 中真正使用它。

它的意义是：

- 过滤掉扫描段里过于不稳定、噪声太大的障碍物片段
- 让后续的障碍物摘要和 FSM 输入更稳

复件里这一层更弱，原文件则把它做成了可调项。

## 8. 控制流程总览

原文件的主链路大致是：

1. `middle_line_callback()` 收到 `LaserScan`
2. `frontend_scan_process()` 做障碍物分段、动态障碍判断、净空计算
3. `heuristic_backend_control()` 先给出一个基础转向和速度
4. `_update_behavior_state()` 根据障碍物和净空更新行为状态
5. `build_local_ref()` 和 `solve_mpc()` 在行为约束下优化控制
6. `control_select_and_publish()` 发布最终驱动、调试和诊断话题

## 9. 一句话结论

`battle_fast2_node.py` 不是简单的“另一个复件”，而是一个带有：

- 行为 FSM
- 动态障碍物跟踪
- 风险感知 MPC
- 行为驱动速度/航向调制
- 更完整诊断话题

的增强版节点。
