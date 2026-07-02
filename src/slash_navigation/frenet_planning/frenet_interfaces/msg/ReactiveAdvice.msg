# Inherit input LaserScan header.
std_msgs/Header header

# When false, all bool fields are false, numeric fields are 0, and consumers must ignore the frame.
bool valid

# Whether the front sector is below the blocking distance threshold.
bool front_blocked

# Whether the left/right side has a traversable gap satisfying minimum clearance and width.
bool left_gap_available
bool right_gap_available

# Whether the left/right side has a gap wide enough for overtaking.
bool overtake_left_available
bool overtake_right_available

# Estimated front/left/right clearances in meters.
float64 front_clearance_m
float64 left_clearance_m
float64 right_clearance_m

# Best gap angle relative to vehicle forward direction; left positive, right negative, radians.
float64 best_gap_angle_rad

# Best gap angular width in radians.
float64 best_gap_width_rad

# Best gap confidence in [0, 1].
float64 best_gap_confidence

# Dynamic obstacle confidence in [0, 1].
float64 dynamic_obstacle_confidence

# Combined static/dynamic blocking risk in [0, 1].
float64 obstacle_risk

# Number of obstacle candidates detected in the current frame.
uint32 obstacle_count
