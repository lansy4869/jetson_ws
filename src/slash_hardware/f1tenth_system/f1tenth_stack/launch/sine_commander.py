#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

class ResponseAnalyzer(Node):
    def __init__(self):
        super().__init__('response_analyzer')
        
        # 数据缓冲区
        self.cmd_speeds = deque(maxlen=500)
        self.motor_commands = deque(maxlen=500)
        self.actual_speeds = deque(maxlen=500)
        self.timestamps = deque(maxlen=500)
        
        # 订阅话题
        self.cmd_sub = self.create_subscription(
            AckermannDriveStamped, '/ackermann_cmd', self.cmd_callback, 10)
        self.motor_sub = self.create_subscription(
            Float64, '/commands/motor/speed', self.motor_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        self.start_time = time.time()
        
    def cmd_callback(self, msg):
        current_time = time.time() - self.start_time
        self.cmd_speeds.append(msg.drive.speed)
        self.timestamps.append(current_time)
        
    def motor_callback(self, msg):
        self.motor_commands.append(msg.data)
        
    def odom_callback(self, msg):
        actual_speed = msg.twist.twist.linear.x
        self.actual_speeds.append(actual_speed)
        
        # 实时显示数据（可选）
        if len(self.cmd_speeds) > 10:
            cmd_current = self.cmd_speeds[-1]
            motor_current = self.motor_commands[-1] if len(self.motor_commands) > 0 else 0
            actual_current = actual_speed
            
            self.get_logger().info(
                f'Cmd: {cmd_current:.2f} | Motor: {motor_current:.2f} | Actual: {actual_current:.2f}',
                throttle_duration_sec=1.0
            )

def main():
    rclpy.init()
    analyzer = ResponseAnalyzer()
    
    try:
        rclpy.spin(analyzer)
    except KeyboardInterrupt:
        # 绘制最终结果
        if len(analyzer.cmd_speeds) > 10:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(analyzer.timestamps, analyzer.cmd_speeds, 'b-', label='命令速度', linewidth=2)
            if len(analyzer.motor_commands) == len(analyzer.cmd_speeds):
                plt.plot(analyzer.timestamps, analyzer.motor_commands, 'g--', label='电机命令', linewidth=2)
            if len(analyzer.actual_speeds) == len(analyzer.cmd_speeds):
                plt.plot(analyzer.timestamps, analyzer.actual_speeds, 'r-', label='实际速度', linewidth=2)
            
            plt.ylabel('速度 (m/s)')
            plt.title('底盘正弦响应测试')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('/tmp/chassis_response.png')
            print("分析图已保存至 /tmp/chassis_response.png")
            
    finally:
        analyzer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()