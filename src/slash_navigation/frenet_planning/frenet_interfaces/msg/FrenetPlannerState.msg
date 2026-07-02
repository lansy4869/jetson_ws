# Planner behavior states.
uint8 GB_TRACK=0
uint8 FOLLOW=1
uint8 OVERTAKE_LEFT=2
uint8 OVERTAKE_RIGHT=3
uint8 RETURN_TO_CENTER=4
uint8 EMERGENCY=5

std_msgs/Header header

# false means the planner state should not be consumed for control.
bool valid

# One of the state constants above.
uint8 state

# Desired lateral offset in the Frenet frame, meters. Left is positive.
float64 target_d

# Desired trajectory speed, meters per second.
float64 target_speed
