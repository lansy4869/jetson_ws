# 继承输入 Odometry 的 header；stamp 为原始 odometry 观测/发布时刻，
# frame_id 是输入全局参考系（通常为 map），不是 Frenet 坐标系名称。
std_msgs/Header header

# valid=false 时，reinitialized=false 且所有数值字段为零。
bool valid
# 仅在 valid=true 时有效；首次有效全局投影或近邻投影回退全局投影成功时为 true；
# 这是单帧脉冲，下一条正常近邻投影必须为 false；下游应清除依赖上一帧 s_unwrapped 的历史。
bool reinitialized

# s 为沿赛道弧长，范围 [0, reference_length)，单位米。
float64 s
# s_unwrapped 为连续弧长，单位米；仅 reinitialized=true 时允许跳变。
float64 s_unwrapped

# d 左正右负，单位米。
float64 d

# yaw_error 为车身航向相对切线的误差，已归一化到 [-pi, pi]，单位 rad。
float64 yaw_error

# speed 直接来自 Odometry.twist.twist.linear.x，是车体 x 轴的带符号纵向速度，单位 m/s。
float64 speed
