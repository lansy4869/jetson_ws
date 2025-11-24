import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64 # 消息类型
import math

class SignalGenerator(Node): 

    def __init__(self):
        super().__init__('signal_generator')
        
        # 创建发布者（Publisher）
        # 发布到 '/commands/motor/speed' 话题
        # 消息类型为 Float64
        # 队列大小设为 10
        self.publisher_ = self.create_publisher(Float64, '/commands/motor/speed', 10) # 注意2：话题名需确认

        # 设置定时器周期为0.01秒（100Hz每0.01秒会调用一次 timer_callback 函数
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # 初始化一个计数器 i，用于计算时间
        self.i = 0

        # **************** 用户可配置参数 **************** 
        self.signal_type = 'sine'  # 信号类型：'sine' 或 'step'
        self.amplitude = 1000.0    # 信号幅值 (对于阶跃信号，这就是阶跃后的值)。单位通常是RPM。
        self.frequency = 0.5       # 正弦信号的频率 (Hz)
        self.offset = 0.0          # 信号的直流偏移量。例如，设为500，则正弦会在500上下波动。
        self.step_start_time = 5.0 # 阶跃信号开始发生的时间（单位：秒）
        # ***********************************************

    def timer_callback(self):
        # 每次定时器触发时，这个函数都会被调用
        msg = Float64() # 创建一个Float类型的消息对象

        # 计算当前时间（单位：秒）
        # self.i 是回调次数，乘以周期0.01s得到时间
        current_time = self.i * 0.01

        # 根据选择的信号类型生成不同的信号
        if self.signal_type == 'sine':
            # 生成正弦波: offset + amplitude * sin(2 * π * frequency * time)
            msg.data = self.offset + self.amplitude * math.sin(2 * math.pi * self.frequency * current_time)
        
        elif self.signal_type == 'step':
            # 生成阶跃波
            if current_time < self.step_start_time:
                msg.data = self.offset # 阶跃发生之前，输出偏移量（通常是0）
            else:
                msg.data = self.offset + self.amplitude # 阶跃发生后，输出偏移量+幅值
        
        else:
            # 如果是其他未知类型，安全起见发送0
            msg.data = 0.0

        # 发布生成的消息到指定话题
        self.publisher_.publish(msg)
        # 在日志中打印发布的数据，便于实时监控
        self.get_logger().info('Publishing: "%s"' % msg.data)
        
        # 计数器加1
        self.i += 1

def main(args=None):
    # 初始化ROS2客户端库
    rclpy.init(args=args)
    # 创建SignalGenerator节点对象
    signal_generator = SignalGenerator()
    # 保持节点运行，等待回调触发
    rclpy.spin(signal_generator)
    # 关闭节点
    signal_generator.destroy_node()
    # 关闭ROS2
    rclpy.shutdown()

if __name__ == '__main__':
    main()