#!/usr/bin/env python3
import math
import cvxpy
from dataclasses import dataclass, field
import numpy as np

import tf2_ros
import rclpy
import rclpy.duration
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, Vector3, PointStamped, Point
from nav_msgs.msg import Odometry
from numpy.matlib import repmat
import rclpy.time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import TransformStamped
from tf2_geometry_msgs import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, Vector3Stamped
from visualization_msgs.msg import Marker, MarkerArray
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
import numpy.linalg as LA
import transforms3d
from tf_transformations import quaternion_matrix, euler_from_quaternion
from rclpy.qos import ReliabilityPolicy, QoSProfile
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import LaserScan
import tf_transformations
import os
from ament_index_python import get_package_share_directory


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0

@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 13 # finite time horizon length kinematic
    iteration = 1

    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 100]) #([0.01, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 100]) #([0.01, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0]) # ([13.5, 13.5, 5.5, 13.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0]) # ([13.5, 13.5, 5.5, 13.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.3  # Length of the vehicle [m]
    WIDTH: float = 0.2  # Width of the vehicle [m]
    WB: float = 0.25  # Wheelbase [m]
    MIN_STEER: float = -math.radians(45)  # maximum steering angle [rad]
    MAX_STEER: float = math.radians(45)  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 2.0  # maximum speed [m/s]
    MIN_SPEED: float = -2.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 2.0  # maximum acceleration [m/ss]
    MAX_DECEL: float = -2.0

class MPC(Node):
    def __init__(self):
        super().__init__('mpc_node')

        self.car_odom_ = self.create_subscription(Odometry,"/Odometry", self.odom_callback, 10)
        self.car_drive_pub_ = self.create_publisher(AckermannDriveStamped,"/drive",10)
        self.controller = self.create_timer(0.1, self.timer_callback)

        self.ref_pub_ = self.create_publisher(Marker, '/ref_trajectory', 10)
        self.predict_pub_ = self.create_publisher(Marker, '/predict_trajectory', 10)
        self.waypoint_pub_ = self.create_publisher(MarkerArray, '/waypoints', 10)

        self.config = mpc_config()
        self.mpc_prob_init()

        self.have_odom_ = False
        self.car = State()
        self.opt_a = [0.0] * self.config.TK
        self.opt_d = [0.0] * self.config.TK

        self.load_waypoints_success = False
        project_file = os.path.join(get_package_share_directory('simple_mpc'), 'waypoints', 'test_waypoints.csv')
        self.load_waypoints(project_file)

        self.nCount = 100
        self.waypoints = self.create_waypoints()


    def mpc_prob_init(self):
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )
        # Control Input vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))
 

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # --------------------------------------------------------
        # TODO: fill in the objectives here, you should be using cvxpy.quad_form() somehwhere

        # TODO: Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)
        
        # TODO: Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)

        # TODO: Objective part 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis=1)), Rd_block)
        
        

        # --------------------------------------------------------

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []

        # 初始化路径为零
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))

        # 迭代进行状态更新
        for t in range(self.config.TK):
            # 使用之前迭代的 delta（即 path_predict[3, t]）
            delta = path_predict[3, t]  # 获取当前状态的航向角作为 delta
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], delta  # 使用更新的 delta
            )
            
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        # 将 A_block, B_block 和 C_block 转换为块对角矩阵
        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        # self.get_logger().info("初始化的A")
        # # 使用 self.get_logger().info 打印调试信息
        # self.get_logger().info(f"A_block shape: {A_block.shape}")
        # self.get_logger().info(f"A_block nnz: {A_block.nnz}")
        # self.get_logger().info(f"A_block data shape: {A_block.data.shape}")
        

        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        self.get_logger().info(f"self.Annz_k shape: {self.Annz_k.shape}")
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------------
        # TODO: Constraint part 1:
        #       Add dynamics constraints to the optimization problem
        #       This constraint should be based on a few variables:
        #       self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_
        constraints += [cvxpy.vec(self.xk[:, 1:]) == self.Ak_ @ cvxpy.vec(self.xk[:, :-1]) +
                        self.Bk_ @ cvxpy.vec(self.uk) + self.Ck_]
        
        # TODO: Constraint part 2:
        #       Add constraints on steering, change in steering angle
        #       cannot exceed steering angle speed limit. Should be based on:
        #       self.uk, self.config.MAX_DSTEER, self.config.DTK
        constraints += [self.xk[:, 0] == self.x0k]

        # TODO: Constraint part 3:
        #       Add constraints on upper and lower bounds of states and inputs
        #       and initial state constraint, should be based on:
        #       self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        #       self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER
        state_constraints, input_constraints, input_diff_constraint = self.get_model_constraints()
        for i in range(4):
            constraints += [state_constraints[0, i] <= self.xk[i, 1:],
                            self.xk[i, 1:] <= state_constraints[1, i]]

        for i in range(2):
            constraints += [input_constraints[0, i] <= self.uk[i, :],
                            self.uk[i, :] <= input_constraints[1, i]]
            # constraints += [input_diff_constraint[0, i] <= cvxpy.diff(self.uk[i, :]),
            #                 cvxpy.diff(self.uk[i, :]) <= input_diff_constraint[1, i]]
        
        # constraints += [self.xk[2,-1] == 0]
        
        # -------------------------------------------------------------

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def get_model_matrix(self, v, phi, delta):
        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def get_model_constraints(self):
        #  """
        # Lower and upper constraints for the state, input and input derivative
        # """
        # state = [x, y, v, yaw]
        state_constraints = np.array([[-1e5, -1e5, self.config.MIN_SPEED, -1e5],
                                      [1e5, 1e5, self.config.MAX_SPEED, 1e5]])
        # input = [acceleration, steering angle]
        input_constraints = np.array([[self.config.MAX_DECEL, self.config.MIN_STEER],
                                      [self.config.MAX_ACCEL, self.config.MAX_STEER]])

        input_diff_constraints = np.array([[-1e5, -self.config.MAX_DSTEER * self.config.DTK],
                                           [1e5, self.config.MAX_DSTEER * self.config.DTK]])
        return state_constraints, input_constraints, input_diff_constraints
    
    def load_waypoints(self, file_path):
        self.center = np.genfromtxt(file_path, delimiter=",")[:,:2]
        self.checkpoint_V = 4*np.ones(len(self.center))
        x = self.center[:,0]
        y = self.center[:,1]
        x_diff = np.diff(x)
        x_diff = np.append(x_diff,x_diff[-1])
        y_diff = np.diff(y)
        y_diff = np.append(y_diff,y_diff[-1])
        self.checkpoint_yaw = np.arctan2(y_diff, x_diff)
        self.load_waypoints_success = True

    def odom_callback(self, msg: Odometry):
        self.have_odom_ = True

        if (self.nCount  == 100):
            self.get_logger().info("接收到odom")
            self.nCount = 0        
        self.nCount = self.nCount+1
        self.car.x = msg.pose.pose._position.x
        self.car.y = msg.pose.pose._position.y
        self.car.v = msg.twist.twist.linear.x
        
        quat = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        euler= tf_transformations.euler_from_quaternion(quat)
        self.car.yaw = euler[2]

    def timer_callback(self):
        if (self.have_odom_==False):
            return
        if (self.load_waypoints_success==False):
            return
        
        ref_x = self.center[:,0]
        ref_y = self.center[:,1]
        ref_yaw = self.checkpoint_yaw
        ref_v = self.checkpoint_V
        ref_path = self.calc_ref_trajectory(ref_x, ref_y, ref_yaw, ref_v)
        self.publish_ref_traj(ref_path)
        self.publish_waypoints()

        (
            self.opt_a,
            self.opt_d,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path)
        self.publish_predict_traj(ox,oy)

        steer_output = self.opt_d[0]
        speed_output = self.car.v + self.opt_a[0] * self.config.DTK
        if speed_output > 0.5:
            speed_output = 0.5
        driver_cmd = AckermannDriveStamped()
        driver_cmd.drive.steering_angle = steer_output
        driver_cmd.drive.speed = speed_output
        self.car_drive_pub_.publish(driver_cmd)



    def publish_predict_traj(self, predict_traj_x, predict_traj_y):
        """
        ref_traj: (x, y, v, yaw) in body frame
        """
        predict = Marker(type=Marker.LINE_STRIP,
                        scale=Vector3(x=0.1, y=0.1, z=0.1))
        predict.header.frame_id = 'lidar_odom'
        predict.color.r = 1.0
        predict.color.g = 1.0
        predict.color.b = 0.0
        predict.color.a = 1.0
        predict.id = 1
        # self.get_logger().info(f'这三：{predict_traj_x.shape}')
        for i in range(len(predict_traj_x)):
            x = predict_traj_x[i]
            y = predict_traj_y[i]
            # x, y = body2world(pose_msg, x, y)
            # print(f'Publishing ref traj x={x}, y={y}')
            predict.points.append(Point(x=x, y=y, z=0.0))
        self.predict_pub_.publish(predict)

        


    def linear_mpc_control(self, ref_path):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        path_predict = self.predict_motion(ref_path)
        x0 = [self.car.x, self.car.y, self.car.v, self.car.yaw]
        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )


        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0



        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], self.opt_d[t]
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)


        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI; cvxpy.mosek
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()
            self.get_logger().info("solve success")

        else:
            self.get_logger().info("canot  solve")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov


    def predict_motion(self, ref_path):
        path_predict = ref_path * 0.0
        x0 = [self.car.x, self.car.y, self.car.v, self.car.yaw]
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=self.car.x, y=self.car.y, yaw=self.car.yaw, v=self.car.v)
        for (ai, di, i) in zip(self.opt_a, self.opt_d, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict
    
    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state


    def create_waypoints(self):
        x_skel = []
        y_skel = []
        yaw_skel = []
        v_skel = []
        x_skel = self.center[:,0]
        y_skel = self.center[:,1]
        yaw_skel = self.checkpoint_yaw
        v_skel = self.checkpoint_V
        waypoints = []
        for i in range(len(x_skel)):
            waypoints.append([x_skel[i], y_skel[i], yaw_skel[i], v_skel[i]])
        waypoints = np.array(waypoints)
        return waypoints

    def publish_waypoints(self):
        marker = Marker()
        marker.header.frame_id = 'lidar_odom'
        marker.type = Marker.POINTS
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.id = 0
        marker.points = [Point(x=x, y=y, z=0.0) for x, y, _, _ in self.waypoints]

        marker_array = MarkerArray()
        marker_array.markers = [marker]
        self.waypoint_pub_.publish(marker_array)

    def publish_ref_traj(self, ref_traj):
        target = Marker(type=Marker.LINE_STRIP,
                        scale=Vector3(x=0.1, y=0.1, z=0.1))
        target.header.frame_id = 'lidar_odom'
        target.color.r = 0.0
        target.color.g = 0.0
        target.color.b = 1.0
        target.color.a = 1.0
        target.id = 1
        for i in range(ref_traj.shape[1]):
            x, y, _, _ = ref_traj[:, i]
           
            target.points.append(Point(x=x, y=y, z=0.0))
        self.ref_pub_.publish(target)

    


    def calc_ref_trajectory(self, cx, cy, cyaw, sp):
        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(self.center[:,0])

         # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = self.nearest_point(np.array([self.car.x, self.car.y]), np.array([cx, cy]).T)
        ind += 1
        if ind >= ncourse:
            ind -= ncourse
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        travel = 0.02
        dind = travel / self.config.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]

        cyaw[cyaw - self.car.yaw > np.pi] -= 2*np.pi
        cyaw[cyaw - self.car.yaw < -np.pi] += 2*np.pi
        ref_traj[3, :] = cyaw[ind_list]

        return ref_traj



    def nearest_point(self, point, trajectory):
        diffs = trajectory[1:,:] - trajectory[:-1,:]
        l2s   = diffs[:,0]**2 + diffs[:,1]**2
        dots = np.empty((trajectory.shape[0]-1, ))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
        t = dots / l2s
        t[t<0.0] = 0.0
        t[t>1.0] = 1.0
        projections = trajectory[:-1,:] + (t*diffs.T).T
        dists = np.empty((projections.shape[0],))
        for i in range(dists.shape[0]):
            temp = point - projections[i]
            dists[i] = np.sqrt(np.sum(temp*temp))
        min_dist_segment = np.argmin(dists)
        return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    PNC_node = MPC()
    
    rclpy.spin(PNC_node)
    rclpy.shutdown()
    
    # try:
    #     rclpy.spin(PNC_node)
    # except KeyboardInterrupt:
    #     # 用户按下Ctrl+C时退出
    #     pass
    # finally:
    #     # 节点结束后绘制数据
    #     PNC_node.plot_results()  # 新增绘图调用
    #     PNC_node.destroy_node()
    #     rclpy.shutdown()

if __name__ == "__main__":
    main()


        



