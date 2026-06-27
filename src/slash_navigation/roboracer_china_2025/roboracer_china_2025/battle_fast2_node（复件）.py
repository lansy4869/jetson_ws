#! /usr/bin/env python3
#coding=utf-8

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration
from std_msgs.msg import Bool
import copy

# === 保持原代码的常量定义不变 ===
DIR_DETECT_THRESHOLD = 2.5 # 方向探测距离,用于方向判断
OBS_DETECT_THRESHOLD = 5.0 # 障碍探测距离,用于障碍物判断

MAX_SPEED_RATE = 2.0

THRESHOLD_obs = 0.5
THRESHOLD_TURN = 0.5 # 转弯阈值
START_ANGLE = -60
END_ANGLE = 60
MIN_OBS_SPEED = 0.5

class BattleVehicleNode(Node):
    def __init__(self):
        super().__init__('wall_following2')
        
        # === 【新增】实车参数声明 (参考 battle_fast2_node.py) ===
        # 默认值保持你代码里的原样，但现在可以在 launch 文件里改了
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('marker_topic', '/arrow_marker_02')
        self.declare_parameter('debug_scan_topic', '/front_scan_02')
        # 实车安全/保守化参数（W1）：全局速度比例、速度上限、转向限幅、调试日志开关
        self.declare_parameter('speed_scale', 0.6)   # 上车保守起步：最终速度乘此系数
        self.declare_parameter('max_speed', 2.0)     # m/s，最终速度硬上限
        self.declare_parameter('max_steer', 0.34)    # rad，转向限幅（与实测 servo 行程一致）
        self.declare_parameter('verbose', False)     # True 才打节流日志，默认静默（不刷屏）
        # W4 创新点2：用外部运动学动态判据（obstacle_tracker）替换不可移植的 intensity 判据
        self.declare_parameter('use_external_dynamic', True)         # True: 用 /perception/dynamic_obstacle
        self.declare_parameter('dynamic_flag_topic', '/perception/dynamic_obstacle')

        scan_topic = self.get_parameter('scan_topic').value
        drive_topic = self.get_parameter('drive_topic').value
        marker_topic = self.get_parameter('marker_topic').value
        debug_scan_topic = self.get_parameter('debug_scan_topic').value
        self.speed_scale = float(self.get_parameter('speed_scale').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.max_steer = float(self.get_parameter('max_steer').value)
        self.verbose = bool(self.get_parameter('verbose').value)
        self.use_external_dynamic = bool(self.get_parameter('use_external_dynamic').value)
        self.dynamic_flag_topic = str(self.get_parameter('dynamic_flag_topic').value)
        # 外部动态障碍标志（由 obstacle_tracker 基于多帧运动学/单帧测速给出）
        self._ext_dynamic = False

        self.get_logger().info(
            f"Battle Node Started. Topics: scan={scan_topic}, drive={drive_topic} | "
            f"speed_scale={self.speed_scale} max_speed={self.max_speed} max_steer=±{self.max_steer}")

        # === 将原代码的 global 变量初始化为类成员变量 ===
        self.last_angle = 0
        self.last_max_dir_index = 0
        self.GO_STARIGHT = 0 # 是否进入直道路段
        self.TRANSITION = 0 # 是否进入过渡路段
        self.last_in_normol = False   # 记录上一次循环是否进入隐藏款
        self.last_in_straight = False  # 记录上一次循环是否进入直道路段
        self.speed_rate = 1.0         # 速度比例因子，初始为1.0
        self.straight_cnt = 0
        self.Follow = False
        self.turn_rate = 1.0 # 转向比例因子
        self.P = 1.1
        self.D = 0.2
        self.dynamic_obs = False
        self.chaoche = False

        # === ROS 2 通信配置 ===
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 订阅雷达 (使用参数 scan_topic)
        self.scan_sub = self.create_subscription(
            LaserScan, 
            scan_topic, 
            self.middle_line_callback, 
            qos_profile)
        
        # 发布控制 (使用参数 drive_topic)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, 
            drive_topic, 
            1)
        
        # 调试话题
        self.scan_pub = self.create_publisher(
            LaserScan, 
            debug_scan_topic, 
            10)
        self.marker_pub = self.create_publisher(
            Marker, 
            marker_topic, 
            1)

        # W4：订阅运动学动态障碍标志（obstacle_tracker 输出），替代 intensity 判据
        if self.use_external_dynamic:
            self.dyn_sub = self.create_subscription(
                Bool, self.dynamic_flag_topic, self._on_dynamic_flag, qos_profile)

    def _p(self, *args):
        """替代原 print 的节流调试日志：默认 verbose=False 时完全静默，
        避免 50Hz 实车刷爆终端（遵循 skill 的「禁止 print 刷屏」规约）。"""
        if not self.verbose:
            return
        msg = ' '.join(str(a) for a in args)
        self.get_logger().info(msg, throttle_duration_sec=1.0)

    def _on_dynamic_flag(self, msg):
        self._ext_dynamic = bool(msg.data)

    # === 原样保留的辅助函数 (微调 publish_arrow_marker 适配 Frame ID) ===

    # 【修改】增加 frame_id 参数，不再硬编码 ego_racecar
    def publish_arrow_marker(self, max_dir_index, frame_id="laser"):
        marker = Marker()
        # 适配实车坐标系 (动态获取)
        marker.header.frame_id = frame_id 
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "direction_arrow"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        marker.points = []
        p0 = Point(x=0.0, y=0.0, z=0.0)
        # 计算箭头终点，假定箭头长度为1
        angle_rad = math.radians(max_dir_index)
        p1 = Point(x=math.sin(angle_rad), y=math.cos(angle_rad), z=0.0)
        marker.points.append(p0)
        marker.points.append(p1)
        marker.scale.x = 0.1  
        marker.scale.y = 0.1  
        marker.scale.z = 0.2  
        marker.color.a = 1.0
        marker.color.r = 0.1
        marker.color.g = 1.0  # 绿色
        marker.color.b = 0.1

        marker.lifetime = Duration(sec=0, nanosec=100000000) # 0.1s
        self.marker_pub.publish(marker)

    def get_dis(self, data, angle, deg=True, return_inten = True):
        if deg:
            angle = np.deg2rad(angle)
        dis = 0
        intensities = None
        # ROS 2 中 angle_min 等属性用法一致
        temp = int((angle - data.angle_min) / data.angle_increment)

        # 增加边界检查防止仿真器数据越界崩溃
        start_idx = max(0, temp-2)
        end_idx = min(len(data.ranges), temp+2)
        
        data_tmp = data.ranges[start_idx:end_idx]
        inten_tmp = data.intensities[start_idx:end_idx] if len(data.intensities) > 0 else [0]*len(data_tmp)
        
        # 必须转为 numpy 才能 sort
        data_tmp = np.sort(np.array(data_tmp))
        inten_tmp = np.sort(np.array(inten_tmp))

        # 保持原逻辑：取第2个元素（中位数/3rd）
        if len(data_tmp) > 2:
            dis = data_tmp[2]
            intensities = inten_tmp[2]
        elif len(data_tmp) > 0:
            dis = data_tmp[0]
            intensities = inten_tmp[0]
            
        if return_inten:
            return dis,intensities
        else:
            return dis

    # 严格保留原逻辑的循环调用方式
    def get_range(self, data, start_angle, end_engle, return_inten = False):
        all_dis = []
        all_inten = []
        for angle in range(start_angle,end_engle):
            tmp = self.get_dis(data, angle, return_inten = return_inten)
            all_dis.append(tmp[0])
            all_inten.append(tmp[1])
        if return_inten:
            return all_dis,all_inten
        else:
            return all_dis

    def fill_zeros_with_neighbors(self, data):
        result = list(data) # 确保复制
        n = len(result)

        for i in range(n):
            if result[i] == 0:
                # Try to use the left neighbor
                left = next((result[j] for j in range(i - 1, -1, -1) if result[j] != 0), None)
                if left is not None:
                    result[i] = left
                    continue

                # Otherwise try to use the right neighbor
                right = next((result[j] for j in range(i + 1, n) if result[j] != 0), None)
                if right is not None:
                    result[i] = right
                    continue

                result[i] = 0
                # self._p("Warning: No non-zero neighbors found for index", i) # 保持注释或打印
        
        return result

    def filter_obstacles_by_variance(self, Left_obs_orig, dis_90, variance_threshold=1.0):
        Left_obs = []  
        if len(Left_obs_orig) > 0:
            for i in range(0, int(len(Left_obs_orig) / 2), 1):
                idx_start = int(Left_obs_orig[2 * i])
                idx_end = int(Left_obs_orig[2 * i + 1])
                
                # dis_obs_middle = dis_90[int((idx_start + idx_end) / 2)]
                obstacle_range = dis_90[idx_start: idx_end]
                dis_obs_var = np.var(obstacle_range)

                self._p("in filter_obstacles_by_variance,方差：",dis_obs_var)
                Left_obs.append(idx_start)
                Left_obs.append(idx_end)
        return Left_obs

    def filter_anomalous_values(self, data, max_distance=4, angle_range=2):
        data = np.array(data)
        for i in range(1, len(data) - 1):
            if data[i] != max_distance:
                if all(data[j] == max_distance for j in range(i - angle_range, i + angle_range + 1) if 0 <= j < len(data)):
                    data[i] = (data[i-1] + data[i+1]) / 2
        return data.tolist()

    def filter_small_obstacles(self, Left_obs, min_obstacle_size=2):
        for i in range(int(len(Left_obs) / 2)):
            if abs(Left_obs[2 * i] - Left_obs[2 * i + 1]) <= min_obstacle_size:
                Left_obs[2 * i] = -1  
                Left_obs[2 * i + 1] = -1  
        Left_obs_temp = Left_obs
        Left_obs = [x for x in Left_obs_temp if x != -1]
        return Left_obs

    def pub_scan(self, dis_90, header):
        scan_msg = LaserScan()
        scan_msg.header = header
        scan_msg.angle_min =  np.pi / 2  # 90 degrees
        scan_msg.angle_max = -np.pi / 2    # 90 degrees
        scan_msg.angle_increment =  -np.pi/180  # 1 degree in radians
        scan_msg.ranges = [float(x) for x in dis_90] # 确保是float list
        scan_msg.intensities = [] 
        scan_msg.range_max = 100.0
        self.scan_pub.publish(scan_msg)

    def DynamicObastcle(self, dis_list,inten_list, max_dir_num, obs):
        if len(max_dir_num) < 3: return False # 防止索引越界
        max_range = [max_dir_num[0], max_dir_num[-1]]
        obs_range = [max_dir_num[1], max_dir_num[2]]
        
        range_list = inten_list[max_range[0]:max_range[1]]
        obs_intensity = inten_list[obs_range[0]:obs_range[1]]
        
        # 增加空数组检查
        if len(range_list) == 0 or len(obs_intensity) == 0: return False

        average_obs_intensity = np.mean(obs_intensity)
        average_intensity = np.mean(range_list)
        abs_tmp = abs(average_obs_intensity - average_intensity)
        if abs_tmp > 5:
            self._p('detect dynamic obs abs_tmpabs_tmpabs_tmpabs_tmp!!!!',abs_tmp)
            return True
        else:
            return False 

    # === 核心回调逻辑（完全保持原流程） ===
    def middle_line_callback(self, data):
        # 仿真器兼容性补丁：将 inf 转为 0，因为你的算法(fill_zeros)是专门处理0的
        # 如果不加这行，get_dis 拿到的都是 inf，算法会认为没有数据
        clean_ranges = np.nan_to_num(np.array(data.ranges), posinf=0.0, neginf=0.0)
        data.ranges = clean_ranges.tolist()

        # 【新增】动态获取当前雷达的 Frame ID，不再硬编码 ego_racecar
        current_frame = data.header.frame_id if data.header.frame_id else "laser"

        # 初始化本轮变量
        self.dynamic_obs = False
        self.chaoche = False
        self.Follow = False
        self.D = 0.2
        self._p("###########################################################")
        
        # 1. 调用原来的 get_range 获取数据
        dis_90, inten_90 = self.get_range(data, -89, 91, True)
        dis_90 = dis_90[::-1]
        inten_90 = inten_90[::-1]
        dis_obs_90 = copy.deepcopy(dis_90)
        lenth_dis = len(dis_90)
        
        #方向变量
        left = 0
        right = 0
        Left_obs_orig = []
        Left_obs = []
        max_dis_num = []
        max_dir_num = []
        max_dis = 0 
        max_dir_range = 0
        max_dis_index = 0
        max_dir_index = 0
        
        self.pub_scan(dis_90, data.header)
        
        # 2. 滤波流程 (保持不变)
        dis_90 = self.fill_zeros_with_neighbors(dis_90) 
        inten_90 = self.fill_zeros_with_neighbors(inten_90) 
        dis_obs_90 = self.fill_zeros_with_neighbors(dis_obs_90) 
        
        dis_90_copy = tuple(dis_90)
        
        # 3. 找最大距离 (保持不变)
        for i in range(0,lenth_dis,1):
            if dis_90[i]>max_dis and i>20 and i<160:
                max_dis = dis_90[i]
                max_dis_index = i 
            if  dis_90[i] > DIR_DETECT_THRESHOLD:
                dis_90[i] = DIR_DETECT_THRESHOLD 
            if dis_obs_90[i] > OBS_DETECT_THRESHOLD:
                dis_obs_90[i] = OBS_DETECT_THRESHOLD 
                
        dis_90 = self.filter_anomalous_values(dis_90, max_distance = DIR_DETECT_THRESHOLD, angle_range=2)
        dis_obs_90 = self.filter_anomalous_values(dis_obs_90, max_distance = OBS_DETECT_THRESHOLD, angle_range=2)

        if max_dis_index < 89:
            left = 1
        else:
            right = 1

        # 4. 障碍物提取 (保持不变)
        for i in range(len(dis_90)):
            if dis_90[i]==0:
                pass # self._p("zero is ???????",i)

        for i in range(0,lenth_dis-2,1):
            if dis_obs_90[i]-dis_obs_90[i+1] > THRESHOLD_obs and len(Left_obs_orig)%2 ==0:
                Left_obs_orig.append(i+1)
            elif dis_obs_90[i+1]-dis_obs_90[i] > THRESHOLD_obs and len(Left_obs_orig)%2 ==1:
                Left_obs_orig.append(i)
        
        if len(Left_obs_orig)%2 == 1:
            self._p('error!!!!!!!!!!!!')
            Left_obs_orig.pop()
        
        left_obs_copy = copy.deepcopy(Left_obs_orig)
        Left_obs = self.filter_small_obstacles(Left_obs_orig,min_obstacle_size = 2)
        Left_obs = self.filter_obstacles_by_variance(Left_obs, dis_obs_90, variance_threshold=1.0)
        
        # 5. 障碍膨胀 (保持不变)
        if len(Left_obs)>0 :
            for i in range(0,int(len(Left_obs)/2),1):
                idx_mid = int((Left_obs[2*i]+Left_obs[2*i+1])/2)
                obs_middle = dis_obs_90[idx_mid]
                
                start_expand = int(max(Left_obs[2*i]-min((Left_obs[2*i+1]-Left_obs[2*i])/2*(4-obs_middle),10),0))
                end_expand = int(min(Left_obs[2*i+1]+min((Left_obs[2*i+1]-Left_obs[2*i])/2*(4-obs_middle),10),lenth_dis-1))

                for j in range(start_expand, end_expand, 1):
                    dis_obs_90[j] = obs_middle
                
                Left_obs[2*i]=start_expand
                Left_obs[2*i+1]=end_expand
            self._p("有障碍物，障碍物是",Left_obs)
        else:
            self._p("没有障碍物")

        # 6. 计算可行区域 (保持不变)
        if len(Left_obs)>0 :
            for i in range(0,int(len(Left_obs)/2)+1,1):
                if i == 0:
                    for j in range(Left_obs[0]-1,0,-1):
                        if dis_obs_90[j]<=dis_obs_90[Left_obs[0]+1]:
                            max_dis_num.append(j)
                            max_dis_num.append(Left_obs[0]+1)
                            break
                    if len(max_dis_num)==0:
                        for j in range(0,Left_obs[0]-1,1):
                            if dis_obs_90[j]>=dis_obs_90[Left_obs[0]+1]:
                                max_dis_num.append(j)
                                max_dis_num.append(Left_obs[0]+1)
                                break
                elif i < int(len(Left_obs)/2):
                    max_dis_num.append(Left_obs[2*i-1])
                    max_dis_num.append(Left_obs[2*i])
                elif i == int(len(Left_obs)/2):
                    for j in range(Left_obs[2*i-1]+1,lenth_dis-1,1):
                        if dis_obs_90[j]<=dis_obs_90[Left_obs[2*i-1]-1]:
                            max_dis_num.append(Left_obs[2*i-1]-1)
                            max_dis_num.append(j)
                            break

            max_dis_val = 0
            max_dis_index_temp = max_dis_index
            
            for i in range(0,int(len(max_dis_num)/2),1):
                if max_dis_val < max_dis_num[2*i+1] - max_dis_num[2*i] :
                    max_dis_val = max_dis_num[2*i+1] - max_dis_num[2*i]
                    max_dis_index = (max_dis_num[2*i+1] + max_dis_num[2*i])/2
                if left == 1 and max_dis_index < 90 and dis_obs_90[0]<dis_obs_90[lenth_dis-1]-1:
                    max_dis_index = max_dis_index+5*abs(dis_obs_90[lenth_dis-1]-dis_obs_90[0])
                elif right == 1 and max_dis_index > 90 and dis_obs_90[0]-1>dis_obs_90[lenth_dis-1]:
                    max_dis_index = max_dis_index-5*abs(dis_obs_90[lenth_dis-1]-dis_obs_90[0])

            self._p("max_dis_index_temp",max_dis_index_temp)
            if len(Left_obs) ==2 :
                if max_dis_index_temp >=89:
                    self._p("turn right")
                    if Left_obs[0] >=89:
                        max_dis_index = int((89+Left_obs[0])/2)
                    elif Left_obs[1] <=89:
                        max_dis_index = max_dis_index_temp
                    else:
                        max_dis_index = max_dis_index_temp
                else:
                    if Left_obs[0] >=89:
                        self._p("1")
                        max_dis_index = max_dis_index_temp
                    elif Left_obs[1] <=89:
                        self._p("2")
                        max_dis_index = int((89+Left_obs[1])/2)
                    else:
                        self._p("3")
                        max_dis_index = max_dis_index_temp
            if len(Left_obs) ==4:
                middle_temp = int((Left_obs[1]+Left_obs[2])/2)
                if max_dis_index_temp >=89:
                    if Left_obs[0] >= 89:
                        max_dis_index = int((89+Left_obs[0])/2)
                    elif Left_obs[1] <= 89 and middle_temp >89:
                        max_dis_index = int((Left_obs[1]+max_dis_index_temp)/2)
                    elif middle_temp <= 89 and Left_obs[2] >89:
                        max_dis_index = middle_temp
                    else:
                        max_dis_index = max_dis_index_temp
                else:
                    self._p("turn left")
                    if Left_obs[0] > 89:
                        max_dis_index = max_dis_index_temp
                    elif Left_obs[0] <= 89 and middle_temp >89:
                        max_dis_index = middle_temp
                    elif middle_temp <= 89 and Left_obs[3] >89:
                        max_dis_index = middle_temp
                    else:
                        max_dis_index = int((Left_obs[3] + 89)/2)

        # 7. 寻找最大距离区域 (保持不变)
        for i in range(0,lenth_dis-2,1):
            if dis_90[i]<DIR_DETECT_THRESHOLD and dis_90[i+1] == DIR_DETECT_THRESHOLD and len(max_dir_num)%2==0:
                max_dir_num.append(i+1)
            elif dis_90[i]==DIR_DETECT_THRESHOLD and dis_90[i+1] < DIR_DETECT_THRESHOLD and len(max_dir_num)%2==1:
                max_dir_num.append(i)

        if len(max_dir_num) % 2 == 1 and len(max_dir_num)!= 1:
            self.get_logger().error("出现单个不封闭区域，请检查障碍物检测逻辑")
            for i in range(0,lenth_dis-2,1):
                pass 
    
        # 转弯阶段
        if len(max_dir_num) == 1:
            if max_dir_num[0] < 90:     
                max_dir_index = int((max_dir_num[0])/2)
                self._p("\033[32m转弯阶段1左转，最大距离朝向: %s\033[0m" % (max_dir_index - lenth_dis/2))
            elif max_dir_num[0] > 90:
                max_dir_index = int((max_dir_num[0]+lenth_dis-2)/2)
                self._p("\033[32m转弯阶段1右转，最大距离朝向: %s\033[0m" % (max_dir_index - lenth_dis/2))
            self.GO_STARIGHT = 0
        
        if len(max_dir_num)==2:
            self._p("找到 %d 个最大距离区域，区域大小 %d" % ((int(len(max_dir_num)/2)), max_dir_num[1]-max_dir_num[0]))
            max_dir_index = int((max_dir_num[0]+max_dir_num[1])/2)
            max_dir_range = max_dir_num[1]-max_dir_num[0]

        if len(max_dir_num)>2:
            self._p("有多个最大距离区域，障碍物个数为:",len(Left_obs)/2)

            if len(Left_obs) > 0:
                # W4 创新点2：动态判据来自 obstacle_tracker 的多帧运动学/单帧测速，
                # 而非原来的 intensity 均值差（MID-360 反射率与 2D 雷达 intensity 不可比，不可移植）。
                if self.use_external_dynamic:
                    self.dynamic_obs = bool(self._ext_dynamic)
                else:
                    self.dynamic_obs = False  # 无外部跟踪器时不臆造动态障碍（不再用 intensity）
            
            cand_space = [] 
            cand_dirs = [] 
            for i in range(0,int(len(max_dir_num)/2),1):
                cand_space.append(max_dir_num[2*(i)+1]-max_dir_num[2*(i)])
                cand_dirs.append((max_dir_num[2*(i)+1]+max_dir_num[2*(i)])/2)
            cand_dir_id = np.where(np.array(cand_space)>18)[0] 
            if len(cand_dir_id)!=0:
                selected_dirs = np.array(cand_dirs)[cand_dir_id].tolist()
                max_dir_idx = np.argmin(selected_dirs)
                selected_ranges = np.array(cand_space)[cand_dir_id].tolist()
                max_dir_index = selected_dirs[max_dir_idx]
                max_dir_range = selected_ranges[max_dir_idx]
                cand_dir_chaoche_idx = np.where(np.array(cand_space)>30)[0]
                if self.dynamic_obs:
                    if cand_dir_chaoche_idx.size:
                        self.chaoche = True
                    else:
                        self.Follow = True
            else:
                cand_dir_id = np.argmax(np.array(cand_space))
                max_dir_index = cand_dirs[cand_dir_id]
                max_dir_range = cand_space[cand_dir_id]
                if self.dynamic_obs:
                    if len(max_dir_num) >= 3:
                        max_dir_index = int((max_dir_num[1] + max_dir_num[2]) / 2)
                    max_dir_range = max_dir_num[-1] - max_dir_num[0]
                    self.Follow = True

            if max_dir_index < 90:
                max_dir_index -= 2
            else:
                max_dir_index += 2

        self._p(max_dir_num,self.P)
        if max_dir_index >= 75 and max_dir_index <= 105:
            mean_straight = np.mean(dis_90_copy[80:100])
            self.GO_STARIGHT = 1
            self.TRANSITION = 0      
            if self.last_in_straight and max_dir_range > 20:
                self._p("在直道路段保持加速，最大距离朝向:", (max_dir_index - len(dis_90) / 2)," 加速空间范围：",max_dir_range)  
                self.speed_rate *= 1.35   
                if mean_straight > 11 and len(Left_obs) == 0:
                    limit_rate = 1.8
                elif mean_straight > 8 and len(Left_obs) == 0:
                    limit_rate = 1.5
                elif mean_straight > 7 and len(Left_obs) == 0:
                    limit_rate = 1.4
                else:
                    limit_rate = 1.3
                if self.speed_rate > limit_rate:
                    self.speed_rate = limit_rate
                
                self._p('速度增益', self.speed_rate)
            else:
                self.speed_rate = 1.4  
                
            self.last_in_straight = True    
        elif max_dir_index < 75 and max_dir_index > 0:
            self._p("\033[32m转弯阶段2左转，最大距离朝向: %s\033[0m" % (max_dir_index - lenth_dis/2))
            self.P = 1.5
            self.speed_rate = 1.3
            self.turn_rate = 0.8
            self.last_in_straight = False
        elif max_dir_index > 105:
            self._p("\033[32m转弯阶段2右转，最大距离朝向: %s\033[0m" % (max_dir_index - lenth_dis/2))
            self.P = 1.5
            self.speed_rate = 1.3
            self.turn_rate = 0.8
            self.last_in_straight = False
        
        #过渡路段或转弯路段
        normol = 1
        if len(max_dir_num) == 0:
            if self.GO_STARIGHT == 1 or self.TRANSITION == 1:
                for i in range(0,lenth_dis-2,1):
                    if dis_90[i+1] - dis_90[i] > THRESHOLD_TURN:
                        max_dir_index = int((i+1+len(dis_90)/2)/2)
                        self.get_logger().warn("进入过渡路段，前方左转，最大距离朝向: %f" % (max_dir_index-len(dis_90)/2))
                        self.P = 0.8
                        normol = 0
                    elif dis_90[i] - dis_90[i+1] > THRESHOLD_TURN:
                        max_dir_index = int((i+len(dis_90)/2)/2)
                        self.get_logger().warn("进入过渡路段，前方右转，最大距离朝向: %f" % (max_dir_index-len(dis_90)/2))
                        self.P = 0.8
                        normol = 0
                if normol == 1:
                    max_dir_index = self.last_max_dir_index
                    self._p("\033[38;5;208m隐藏款，无法判断方向，保持上一次动作并减速，最大距离朝向: %f\033[0m" % (max_dir_index-len(dis_90)/2))
                    
                    if self.last_in_normol:
                        self.speed_rate *= 1.2        
                        self.turn_rate *= 1.2
                        if self.speed_rate < 0.9:
                            self.speed_rate = 0.9
                        if self.turn_rate > 2.5:
                            self.turn_rate = 2.5
                    else:
                        self.speed_rate = 1.2         
                        self.turn_rate = 1.2
                    self.last_in_normol = True
                else:
                    self.speed_rate = 1.3             
                    self.turn_rate = 1.0
                    self.last_in_normol = False

                self.TRANSITION = 1
                self.GO_STARIGHT = 0

        dis_90[0] = dis_90[0] + 0.00001
        self._p("视野中最大距离是",max_dis)
        dis_90[lenth_dis-1] = dis_90[lenth_dis-1] +0.00001
        
        angle = 0
        if max_dir_index != 0:
            term1 = -max(math.exp(-max_dis/DIR_DETECT_THRESHOLD),0.7)*(max_dir_index-90)/360 *math.pi
            term2 = (dis_90[0]-dis_90[lenth_dis-1])/(dis_90[0]+dis_90[lenth_dis-1])
            self._p(f'term1:{term1}, term2:{term2}')
            if dis_90[0]/dis_90[lenth_dis-1]>3 or dis_90[lenth_dis-1]/dis_90[0]>3:
                self.D = 0.5
                self._p(f'边界！！！！！！！！！！！！！！！！')
                angle = 1.0 * term1 + 0.05 * term2
            else:
                angle = 1.0 * term1 + 0.02 * term2

        else:
            if dis_90[0]/dis_90[lenth_dis-1]>3 or dis_90[lenth_dis-1]/dis_90[0]>3: 
                angle = -max(math.exp(-max_dis/DIR_DETECT_THRESHOLD),0.7)*(max_dir_index-90)/360 *math.pi + 0.1*(dis_90[0]-dis_90[lenth_dis-1])/(dis_90[0]+dis_90[lenth_dis-1])
            else:
                angle = -max(math.exp(-max_dis/DIR_DETECT_THRESHOLD),0.7)*(max_dir_index-90)/360 *math.pi + 0.05*(dis_90[0]-dis_90[lenth_dis-1])/(dis_90[0]+dis_90[lenth_dis-1]) 

        steering_angle = self.P* angle + self.D *(angle-self.last_angle)
        self.last_angle = angle

        speed = 1.5*(0.3*math.exp(-np.clip(abs(angle),0,0.5))+0.7)
        
        self._p("max_dir_index",max_dir_index)
    
        steering_angle = self.turn_rate*steering_angle
        steering_angle = np.clip(steering_angle, -math.pi/4, math.pi/4)  
        if steering_angle > 0:    
            self._p("转向左转%f度"%(abs(steering_angle*180/math.pi)))
        else:
            self._p("转向右转%f度"%(abs(steering_angle*180/math.pi)))
        
        # 【修改】传入当前帧 Frame ID
        self.publish_arrow_marker(max_dir_index, current_frame)
        self.last_max_dir_index = max_dir_index
    
        drive_msg = AckermannDriveStamped()
        drive_msg.header = data.header # 保持同步
        drive_msg.drive.steering_angle = float(steering_angle)
        drive_msg.drive.speed= float(self.speed_rate*speed)
        
        if self.Follow:
            self._p('\033[35m跟随！！22222！！跟随！！2222！！\033[0m')
            drive_msg.drive.speed = float(min(MIN_OBS_SPEED,self.speed_rate*speed))
        elif self.chaoche:
            drive_msg.drive.speed= float(self.speed_rate*speed)
            self._p("""
            \033[31m   _____                         _____  
            \033[32m  / ____|                       / ____| 
            \033[33m | (___  _   _ _ __   ___ _ __  | |  __ 
            \033[34m  \___ \| | | | '_ \ / _ \ '__| | | |_ |
            \033[35m  ____) | |_| | |_) |  __/ |    | |__| |
            \033[36m |_____/ \__,_| .__/ \___|_|     \_____|
            \033[37m              | |                      
            \033[35m              |_|   \033[5m超车模式启动！！！\033[0m
            """)
            self._p("speed:",drive_msg.drive.speed)

        # === W1 保守化与限幅：全局速度比例 + 速度上限 + 转向限幅（与实测可行域一致）===
        drive_msg.drive.speed = float(min(self.max_speed, self.speed_scale * drive_msg.drive.speed))
        drive_msg.drive.steering_angle = float(np.clip(drive_msg.drive.steering_angle,
                                                       -self.max_steer, self.max_steer))

        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    battle_node = BattleVehicleNode()
    try:
        rclpy.spin(battle_node)
    except KeyboardInterrupt:
        pass
    finally:
        battle_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
