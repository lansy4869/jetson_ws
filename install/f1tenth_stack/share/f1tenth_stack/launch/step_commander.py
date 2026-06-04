#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class StepCommander(Node):
    def __init__(self):
        super().__init__('step_commander')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.start_time = time.time()
        self.step_value = 0.0
        self.step_time = 5.0  # 5秒后阶跃
        
    def timer_callback(self):
        current_time = time.time() - self.start_time
        
        if current_time > self.step_time and self.step_value == 0.0:
            self.step_value = 0.5  # 阶跃到0.5m/s
            
        msg = Twist()
        msg.linear.x = self.step_value
        msg.angular.z = 0.0
        
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: linear.x = {msg.linear.x:.2f}')

def main():
    rclpy.init()
    node = StepCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()