import rclpy
from ackermann_msgs.msg import AckermannDriveStamped

from slash_safety.ggv_shield_node import GgvShieldNode


def test_drive_input_without_fresh_odom_publishes_stop():
    rclpy.init()
    node = GgvShieldNode()
    try:
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = 0.1
        msg.drive.speed = 1.0

        node._on_drive_in(msg)

        assert node.last_out_speed == 0.0
        assert node.last_steer == 0.0
    finally:
        node.destroy_node()
        rclpy.shutdown()
