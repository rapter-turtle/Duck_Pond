#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64MultiArray
from heron_msgs.msg import Drive
from mpc_msgs.msg import MPCTraj, MPC_State, Obs_State
import math
import time
import numpy as np
from l_adaptive import*
from gen_ref import *

traj_xy = (353126.1653380032, 4026065.6140236068)
offset = np.array([353152, 4026032])
class L1_LQR:
    def __init__(self):
        # ROS settings
        self.rate = rospy.Rate(10)  # 10 Hz for ROS loop rate
        self.ekf_sub = rospy.Subscriber('/ekf/estimated_state', Float64MultiArray, self.ekf_callback)
        self.thrust_pub = rospy.Publisher('/cmd_drive', Drive)
        self.estim_pub = rospy.Publisher('/l1_data', Float64MultiArray)
        self.mpcvis_pub = rospy.Publisher('/mpc_vis', MPCTraj)  
        
        # Initial states and inputs
        self.x = self.y = self.p = self.u = self.v = self.r = 0.0
        self.n1 = self.n2 = 0.0
        self.states = np.zeros(8)
        
        self.param_estim = np.array([0.0, 0.0, 0.0, 0.0])
        self.param_filtered = np.array([0.0, 0.0, 0.0, 0.0])
        self.state_estim = np.array([0.0, 0.0,0.0, 0.0])
        self.L1_thruster = np.array([0.0,0.0])

        self.con_dt = 0.1 # control sampling time
        self.ii = 0.0

        self.deried_traj_straight = np.array([0.0, 0.0,0.0, 0.0])
        self.deried_traj_circle = np.array([0.0, 0.0,0.0, 0.0])

        # self.station_keeping_point = np.array([353148, 4026039])
        self.station_keeping_point = np.array([353152, 4026032])
  
          # reference trajectory generation
        self.ref_dt = 0.01
        self.ref_iter = int(self.con_dt/self.ref_dt)
        self.reference = generate_figure_eight_trajectory(1000, self.ref_dt, traj_xy, np.pi/2) # reference dt = 0.01 sec, 1000 sec trajectory generation
        # self.reference = generate_figure_eight_trajectory_con(1000, self.ref_dt) # reference dt = 0.01 sec, 1000 sec trajectory generation
        self.reference = self.reference[::self.ref_iter,:]
        np.save('ref_data.npy', self.reference)


    
    def force_to_heron_command(self,force,left_right):
        """Convert force to Heron command, accounting for dead zones and offsets."""
        # left_right = 0 -> left, 1 -> right
        # dead zone -0.4~0.2
        # param : 25/0.6 (forward), 8/0.6 (backward)
        if force >= 0:
            n = force * 0.6/25
            n += 0.2
            if left_right == 1:
                n += 0.1 # offset left += 0.05
            # if n> 0.8:
            #     n = 0.8            
        else:
            n = force * 0.6/8         
            n -= 0.4
            # if n< -0.8:
            #     n = -0.8            
        return n
    
    def ekf_callback(self, msg):# - frequency = gps callback freq. 
        """Callback to update states from EKF estimated state."""
        self.x, self.y, self.p, self.u, self.v, self.r = msg.data[:6]
              
        # to path fixed coordinate
        heron_pos = np.array([self.x, self.y])
        cte = (heron_pos - self.station_keeping_point)

        self.states = np.array([cte[0], cte[1], self.p, self.u, self.v, self.r, self.n1, self.n2])
        # print(self.p)
        


    def yaw_discontinuity(self, ref):
        """Handle yaw angle discontinuities."""
        flag = [0.0] * 3
        flag[0] = abs(self.states[2] - ref)
        flag[1] = abs(self.states[2] - (ref - 2 * math.pi))
        flag[2] = abs(self.states[2] - (ref + 2 * math.pi))
        min_element_index = flag.index(min(flag))

        if min_element_index == 0:
            ref = ref
        elif min_element_index == 1:
            ref = ref - 2 * math.pi
        elif min_element_index == 2:
            ref = ref + 2 * math.pi
        return ref



    def run(self):
        k = 0 # -> 현재 시간을 index로 표시 -> 그래야 ref trajectory설정가능(******** todo ********)
        last_time = time.time() 

        while not rospy.is_shutdown():
            # station_keeping_point = np.array([353147, 4026038])
            current_time = time.time()
            self.con_dt = current_time - last_time  # Calculate con_dt
            last_time = current_time  # Update last_time for the next loop
            refs = self.reference[k,:]


            # Go straight
            vx = -0.3
            vy = 0.5
            self.desired_traj_straight = np.array([self.ii*self.con_dt*vx,self.ii*self.con_dt*vy,vx,vy])
            # print(self.desired_traj_straight)
            # # Go circle
            omega = 0.2
            theta = omega*self.con_dt*self.ii
            R = 3.0
            self.desired_traj_circle = np.array([R*np.cos(theta), R*np.sin(theta), -R*omega*np.sin(theta), R*omega*np.cos(theta)])      

            self.desired_traj_eight = np.array([refs[0],refs[1], refs[2], refs[3]])

            self.state_estim, self.param_estim, self.param_filtered, L1_thruster, x_error = L1_control(self.states, self.state_estim, self.param_filtered, self.con_dt, self.param_estim, self.desired_traj_eight)

            # obtain mpc input
            self.n1 = L1_thruster[0]
            self.n2 = L1_thruster[1]

            # print("state_estimation : ",self.state_estim)
            # print("state_error", x_error)
            # print("param_estimation : ",self.param_estim)
            print("Control input : ", L1_thruster)
            
            # Publish the control inputs (e.g., thrust commands)
            drive_msg = Drive()
            drive_msg.left = self.force_to_heron_command(self.n1,1)
            drive_msg.right = self.force_to_heron_command(self.n2,2)                        
            self.thrust_pub.publish(drive_msg)  

            l1_msg = Float64MultiArray()
            l1_msg.data = [self.state_estim[0], self.state_estim[1], self.state_estim[2], self.state_estim[3],   #4
                            self.param_estim[2], self.param_estim[3], self.param_filtered[2], self.param_filtered[3],  #4
                            # self.station_keeping_point[0], self.station_keeping_point[1], vx, vy] # 4
                            self.station_keeping_point[0], self.station_keeping_point[1], R, omega]
                            # x_error[0], x_error[1], x_error[2], x_error[3]]
            self.estim_pub.publish(l1_msg)

            # mpc_data_stack = MPCTraj()
            # mpc_data_stack.header.stamp = rospy.Time.now()
            # mpc_data_stack.pred_num = 0.0
            # mpc_data_stack.sampling_time = self.con_dt
            # mpc_data_stack.cpu_time = 0.0
            
            # for j in range(self.N+1):
            #     mpc_pred = MPC_State()
            #     mpc_ref = MPC_State()
            #     mpc_pred.x = self.states[0] + np.cos(self.states[2]) + offset[0]
            #     mpc_pred.y = self.states[1] + np.sin(self.states[2]) + offset[1]
            #     mpc_pred.Vx = self.states[3]*np.cos(self.states[2]) - self.states[4]*np.sin(self.states[2]) - self.states[5]*np.sin(self.states[2])
            #     mpc_pred.Vy = self.states[3]*np.sin(self.states[2]) + self.states[4]*np.cos(self.states[2]) + self.states[5]*np.cos(self.states[2])
            #     mpc_pred.nx = self.force_to_heron_command(self.n1,1)
            #     mpc_pred.ny = self.force_to_heron_command(self.n2,2)
            #     mpc_data_stack.state.append(mpc_pred)            
            #     # print(mpc_pred.u)
            #     mpc_ref.x = self.reference[k+j,0] + offset[0]
            #     mpc_ref.y = self.reference[k+j,1] + offset[1]
            #     mpc_ref.Vx = self.reference[k+j,2]
            #     mpc_ref.Vy = self.reference[k+j,3]
            #     mpc_ref.nx = 0.0
            #     mpc_ref.ny = 0.0
            #     mpc_data_stack.ref.append(mpc_ref)   


            
            self.ii = self.ii + 1

            k = k+1
            if k >= len(self.reference):
                k = 0 
            
            self.rate.sleep()


if __name__ == '__main__':
    try:
        rospy.init_node('L1_LQR', anonymous=True)
        station_keeping = L1_LQR()
        rospy.loginfo("Starting L1_LQR control loop.")
        station_keeping.run()
    except rospy.ROSInterruptException:
        pass