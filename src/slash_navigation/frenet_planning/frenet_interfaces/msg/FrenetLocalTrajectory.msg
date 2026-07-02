std_msgs/Header header

# false means the trajectory is not safe to consume for control.
bool valid

# Mirrors FrenetPlannerState.state.
uint8 state

# Planner targets used to generate this trajectory.
float64 target_d
float64 target_speed

FrenetTrajectoryPoint[] points
