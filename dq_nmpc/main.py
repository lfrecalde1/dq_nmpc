#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

import threading

# Libraries of dual-quaternions
from dq_nmpc import dualquat_from_pose_casadi
from dq_nmpc import dualquat_trans_casadi, dualquat_quat_casadi, rotation_casadi, rotation_inverse_casadi, dual_velocity_casadi, velocities_from_twist_casadi
from dq_nmpc import compute_flatness_states
from dq_nmpc import create_ocp_solver
from dq_nmpc import error_dual_aux_casadi

from acados_template import AcadosOcpSolver, AcadosSimSolver

from ament_index_python.packages import get_package_share_path

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import time
import os

# Function to create a dualquaternion, get quaernion and translatation and returns a dualquaternion
dualquat_from_pose = dualquat_from_pose_casadi()

# Function to get the trasnlation from the dualquaternion, input dualquaternion and get a translation expressed as a quaternion [0.0, tx, ty,tz]
get_trans = dualquat_trans_casadi()

# Function to get the quaternion from the dualquaternion, input dualquaternion and get a the orientation quaternions [qw, qx, qy, qz]
get_quat = dualquat_quat_casadi()

# Function that maps linear velocities in the inertial frame and angular velocities in the body frame to both of them in the body frame, this is known as twist using dualquaternions
dual_twist = dual_velocity_casadi()

# Function that maps linear and angular velocites in the body frame to the linear velocity in the inertial frame and the angular velocity still in th body frame
velocity_from_twist = velocities_from_twist_casadi()

# Function that returns a vector from the body frame to the inertial frame
rot = rotation_casadi()

# Function that returns a vector from the inertial frame to the body frame
inverse_rot = rotation_inverse_casadi()

# Function to check for the shorthest path
error_dual_f = error_dual_aux_casadi()

class DQnmpcNode(Node):
    def __init__(self):
        super().__init__('DQNMPC_FINAL')
        # Lets define internal variables
        self.g = 9.81
        self.mQ = (1.27)

        # Inertia Matrix
        self.Jxx = 0.0030
        self.Jyy = 0.0045
        self.Jzz = 0.00159687
        self.J = np.array([[self.Jxx, 0.0, 0.0], [0.0, self.Jyy, 0.0], [0.0, 0.0, self.Jzz]])
        self.L = [self.mQ, self.Jxx, self.Jyy, self.Jzz, self.g]

        # Desired sample time  self.ts, time where we want to init over trajectory t_initial, time for the trajectory t_trajectory
        # Time to go to the init state t_final
        # Initial time to established a stable connection self.initial
        self.ts = 0.01
        t_inital = 2
        t_trajectory = 30
        t_final = 2
        self.initial = 5

        # Initial States dual set zeros
        # Position of the system
        pos_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Linear velocity of the sytem respect to the inertial frame
        vel_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Angular velocity respect to the Body frame
        omega_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Initial Orientation expressed as quaternionn
        quat_0 = np.array([1.0, 0.0, 0.0, 0.0])

        # Auxiliary vector [x, v, q, w], which is used to update the odometry and the states of the system
        self.x_0 = np.hstack((pos_0, vel_0, quat_0, omega_0))

        ## COmpute desired path based on read Odometry
        print("Computing Path")
        self.hd, self.hd_d, self.hd_dd, self.qd, self.w_d, self.f_d, self.M_d, self.t = compute_flatness_states(self.L, self.x_0[0:3], t_inital, t_trajectory, t_final, self.ts, 2, (self.initial + 1)*0.5)
        print("Path Computed")

        # Define odometry publisher for the drone
        self.odom_msg = Odometry()
        self.publisher_odom_ = self.create_publisher(Odometry, "odom", 10)

        # Define odometry publisher for the desired path
        self.ref_msg = Odometry()
        self.publisher_ref_ = self.create_publisher(Odometry, "desired_frame", 10)

        # Definition of the publihser for the desired parth
        self.marker_msg = Marker()
        self.points = None
        self.publisher_ref_trajectory_ = self.create_publisher(Marker, 'desired_path', 10)

        # Definition of the prediction time in secs
        self.t_N = 0.5

        # Definition of the horizon
        self.N = np.arange(0, self.t_N + self.ts, self.ts)
        self.N_prediction = self.N.shape[0]

        # Auxiliar time in order to establoshed a stable conection
        self.t_aux = np.arange(0, t_inital + self.ts, self.ts, dtype=np.double)

        # New Vector to save the states of the drone as a baseline formulation
        self.x = np.zeros((13, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        self.x[:, 0] = self.x_0

        # Init states formulated as dualquaternions
        self.dual_1 = dualquat_from_pose(self.x_0[6], self.x_0[7], self.x_0[8],  self.x_0[9], self.x_0[0], self.x_0[1], self.x_0[2])

        # Init linear velocity in the inertial frame and angular velocity in the body frame
        self.angular_linear_1 = np.array([self.x_0[10], self.x_0[11], self.x_0[12], self.x_0[3], self.x_0[4], self.x_0[5]]) # Angular Body linear Inertial

        # Init Dual Twist
        self.dual_twist_1 = dual_twist(self.angular_linear_1, self.dual_1)

        # Auxiliar vector where we can to save all the information formulated as dualquaternion
        self.X = np.zeros((14, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        self.X[:, 0] = np.array(ca.vertcat(self.dual_1, self.dual_twist_1)).reshape((14, ))

        # Path where we aere going to save the generate file from acados
        path_file_name = "c_generated_code"
        self.path_file = os.path.join(get_package_share_path("dq_nmpc"), path_file_name)

        # path where we are going to save the images of the simulations for instance positions and more
        path_image_file_name = "results"
        self.path_image_file = os.path.join(get_package_share_path("dq_nmpc"), path_image_file_name)

        # Create a thread to run the simulation and viewer
        self.simulation_thread = threading.Thread(target=self.run)
        # Start thread for the simulation
        self.simulation_thread.start()

    def send_odometry(self, dqd):
        # Function that send odometry

        t_d = get_trans(dqd)
        q_d = get_quat(dqd)

        self.odom_msg.header.frame_id = "world"

        self.odom_msg.header.stamp = self.get_clock().now().to_msg()

        self.odom_msg.pose.pose.position.x = float(t_d[1, 0])
        self.odom_msg.pose.pose.position.y = float(t_d[2, 0])
        self.odom_msg.pose.pose.position.z = float(t_d[3, 0])

        self.odom_msg.pose.pose.orientation.x = float(q_d[1, 0])
        self.odom_msg.pose.pose.orientation.y = float(q_d[2, 0])
        self.odom_msg.pose.pose.orientation.z = float(q_d[3, 0])
        self.odom_msg.pose.pose.orientation.w = float(q_d[0, 0])

        # Send Messag
        self.publisher_odom_.publish(self.odom_msg)
        return None 
  
    def send_ref(self, h, q):
        self.ref_msg.header.frame_id = "world"
        self.ref_msg.header.stamp = self.get_clock().now().to_msg()

        self.ref_msg.pose.pose.position.x = h[0]
        self.ref_msg.pose.pose.position.y = h[1]
        self.ref_msg.pose.pose.position.z = h[2]

        self.ref_msg.pose.pose.orientation.x = q[1]
        self.ref_msg.pose.pose.orientation.y = q[2]
        self.ref_msg.pose.pose.orientation.z = q[3]
        self.ref_msg.pose.pose.orientation.w = q[0]

        # Send Message
        self.publisher_ref_.publish(self.ref_msg)
        return None 

    def init_marker(self, x):
        self.marker_msg.header.frame_id = "world"
        self.marker_msg.header.stamp = self.get_clock().now().to_msg()
        self.marker_msg.ns = "trajectory"
        self.marker_msg.id = 0
        self.marker_msg.type = Marker.LINE_STRIP
        self.marker_msg.action = Marker.ADD
        self.marker_msg.pose.orientation.w = 1.0
        self.marker_msg.scale.x = 0.02  # Line width
        self.marker_msg.color.a = 1.0  # Alpha
        self.marker_msg.color.r = 0.0  # Red
        self.marker_msg.color.g = 1.0  # Green
        self.marker_msg.color.b = 0.0  # Blue
        point = Point()
        point.x = x[0]
        point.y = x[1]
        point.z = x[2]
        self.points = [point]
        self.marker_msg.points = self.points
        return None

    def send_marker(self, x):
        self.marker_msg.header.stamp = self.get_clock().now().to_msg()
        self.marker_msg.type = Marker.LINE_STRIP
        self.marker_msg.action = Marker.ADD
        point = Point()
        point.x = x[0]
        point.y = x[1]
        point.z = x[2]
        self.points.append(point)
        self.marker_msg.points = self.points
        self.publisher_ref_trajectory_.publish(self.marker_msg)
        return None

    def run(self):
        # Vector empty control actions
        F = np.zeros((1, self.t.shape[0] - self.N_prediction), dtype=np.double)
        M = np.zeros((3, self.t.shape[0] - self.N_prediction), dtype=np.double)

        # Generalized control actions
        u = np.zeros((4, self.t.shape[0]-self.N_prediction), dtype=np.double)
        u[0, :] = self.mQ*self.g

        # Constraints on control actions
        F_max = self.mQ*self.g + 20
        F_min = 0.0
        tau_1_max = 0.1
        tau_1_min = -0.1
        tau_2_max = 0.1
        tau_2_min = -0.1
        tau_3_max = 0.1
        taux_3_min = -0.1

        # Empty vector for the desired states defined in dualquaternions
        X_d = np.zeros((14, self.t.shape[0]+1), dtype=np.double)

        # Empty vector for the control actions
        u_d = np.zeros((4, self.t.shape[0]), dtype=np.double)
        for k in range(0, self.t.shape[0]):
            # Define control actions computed previously
            u_d[0, k] = self.f_d[0, k]
            u_d[1, k] = self.M_d[0, k]
            u_d[2, k] = self.M_d[1, k]
            u_d[3, k] = self.M_d[2, k]

            # Desired Orientation
            qw1_d = self.qd[0, k]
            qx1_d = self.qd[1, k]
            qy1_d = self.qd[2, k]
            qz1_d = self.qd[3, k]
            tx1_d = self.hd[0, k]
            ty1_d = self.hd[1, k]
            tz1_d = self.hd[2, k]

            # Desired dualquaternions
            dual_1_d = dualquat_from_pose(qw1_d, qx1_d, qy1_d,  qz1_d, tx1_d, ty1_d, tz1_d)

            # Linear Velocities Inertial frame
            hxd_d = self.hd_d[0, k]
            hyd_d = self.hd_d[1, k]
            hzd_d = self.hd_d[2, k]

            # Angular velocites body frame
            wx_d = self.w_d[0, k]
            wy_d = self.w_d[1, k]
            wz_d = self.w_d[2, k]

            # We do not update the velocites as a dualtwist; instead we use just the 
            # Inertial velocities in inertial frame and the body frame
            angular_linear_1_d = np.array([wx_d, wy_d, wz_d, hxd_d, hyd_d, hzd_d]) # Angular Body linear Inertial

            # Update Reference
            X_d[8:14, k] = angular_linear_1_d
            X_d[0:8, k] = np.array(dual_1_d).reshape((8, ))

        # Init System in order to get stable communication
        for k in range(0, self.t_aux.shape[0]):
            tic = time.time()
            self.x[:, 0] = self.x_0
            self.X[:, 0] = np.array(ca.vertcat(self.dual_1, self.dual_twist_1)).reshape((14, ))
            while (time.time() - tic <= self.ts):
                pass
        
        # Optimization set up
        ocp = create_ocp_solver(self.X[:, 0], self.N_prediction, self.t_N, F_max, F_min, tau_1_max, tau_1_min, tau_2_max, tau_2_min, tau_3_max, taux_3_min, self.L, self.ts, self.path_file)

        # No Cython
        json_name = "acados_ocp_" + ocp.model.name + ".json"
        json_name = str(get_package_share_path("dual_nmpc") / json_name)
        #acados_ocp_solver = AcadosOcpSolver(ocp, json_file=json_name, build= True, generate= True)
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file=json_name, build= False, generate= False)
        #acados_integrator = AcadosSimSolver(ocp, json_file="acados_sim_" + json_name, build= True, generate= True)
        acados_integrator = AcadosSimSolver(ocp, json_file="acados_sim_" + json_name, build= False, generate= False)

        # Reset Solver
        acados_ocp_solver.reset()

        # Initial States Acados
        for stage in range(self.N_prediction + 1):
            acados_ocp_solver.set(stage, "x", self.X[:, 0])
        for stage in range(self.N_prediction):
            acados_ocp_solver.set(stage, "u", u_d[:, 0])

        # Check shorthest path dualquaternion
        error_dual_no_filter = np.array(error_dual_f(X_d[0:8, 0], self.X[0:8, 0])).reshape((8, ))
        if error_dual_no_filter[0] > 0.0:
            X_d[0:8, :] = X_d[0:8, :]
        else:
            X_d[0:8, :] = -X_d[0:8, :]

        # Init Markers
        self.init_marker(self.hd[:, 0])

        # Simulation loop
        for k in range(0, self.t.shape[0] - self.N_prediction):
            # Get model
            tic = time.time()

            # Send Desired States
            self.send_marker(self.hd[:, k])
            self.send_ref(self.hd[:, k], self.qd[:, k])
            self.send_odometry(self.X[0:8, k])

            ## Optimal control setting parameters
            acados_ocp_solver.set(0, "lbx", self.X[:, k])
            acados_ocp_solver.set(0, "ubx", self.X[:, k])

            # Desired Trajectory of the system
            for j in range(self.N_prediction):
                yref = X_d[:,k+j]
                uref = u_d[:,k+j]
                aux_ref = np.hstack((yref, uref))
                acados_ocp_solver.set(j, "p", aux_ref)

            # Desired Trayectory at the last Horizon
            yref_N = X_d[:,k+self.N_prediction]
            uref_N = u_d[:,k+self.N_prediction]
            aux_ref_N = np.hstack((yref_N, uref_N))
            acados_ocp_solver.set(self.N_prediction, "p", aux_ref_N)

            # Check Solution since there can be possible errors 
            #acados_ocp_solver.options_set("rti_phase", 2)
            acados_ocp_solver.solve()
            u[:, k] = acados_ocp_solver.get(0, "u")

            # Send Control actions
            F[:, k] = u[0, k]
            M[:, k] = u[1:4, k]

            # Section to guarantee same sample times
            while (time.time() - tic <= self.ts):
                pass
            toc = time.time() - tic

            # System evolution
            # Update Data of the system
            acados_integrator.set("x", self.X[:, k])
            acados_integrator.set("u", u[:, k])

            status_integral = acados_integrator.solve()
            xcurrent = acados_integrator.get("x")

            # Update Data of the system
            self.X[:, k+1] = xcurrent
            self.get_logger().info("DQ-NMPC CONTROL")
        None
def main(args=None):
    rclpy.init(args=args)
    planning_node = DQnmpcNode()
    try:
        rclpy.spin(planning_node)  # Will run until manually interrupted
    except KeyboardInterrupt:
        planning_node.get_logger().info('Simulation stopped manually.')
        planning_node.destroy_node()
        rclpy.shutdown()
    finally:
        planning_node.destroy_node()
        rclpy.shutdown()
    return None

if __name__ == '__main__':
    main()