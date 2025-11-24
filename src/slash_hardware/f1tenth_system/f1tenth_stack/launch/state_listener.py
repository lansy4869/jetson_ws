#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import json
import time

class StateListener(Node):
    def __init__(self):
        super().__init__('state_listener')
        
        # 订阅命令和实际状态
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
            
        self.cmd_data = []
        self.odom_data = []
        self.start_time = time.time()
        
    def cmd_callback(self, msg):
        current_time = time.time() - self.start_time
        self.cmd_data.append({
            'time': current_time,
            'linear_x': msg.linear.x,
            'angular_z': msg.angular.z
        })
        
    def odom_callback(self, msg):
        current_time = time.time() - self.start_time
        self.odom_data.append({
            'time': current_time,
            'linear_x': msg.twist.twist.linear.x,
            'position_x': msg.pose.pose.position.x
        })
        
    def save_data(self, filename):
        data = {
            'command': self.cmd_data,
            'odometry': self.odom_data
        }
        with open(f'{filename}.json', 'w') as f:
            json.dump(data, f)

def main():
    rclpy.init()
    node = StateListener()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 保存数据
        node.save_data('chassis_response_data')
        print("Data saved to chassis_response_data.json")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()