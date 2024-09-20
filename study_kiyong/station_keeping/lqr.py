import rospy
from std_msgs.msg import Float64MultiArray
from heron_msgs.msg import Drive
# from mpc_msgs.msg import MPCTraj, MPC_State, Obs_State
import math
import time
import numpy as np
from l_adaptive import*

station_keeping_point = np.array([353148, 4026039])

class L1_LQR:
    def __init__(self):
        # ROS settings
        self.rate = rospy.Rate(10)  # 10 Hz for ROS loop rate
        self.ekf_sub = rospy.Subscriber('/ekf/estimated_state', Float64MultiArray, self.ekf_callback)
        self.thrust_pub = rospy.Publisher('/cmd_drive', Drive)
        # self.mpcvis_pub = rospy.Publisher('/mpc_vis', MPCTraj)  
        
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
        else:
            n = force * 0.6/8         
            n -= 0.4
        return n
    
    def ekf_callback(self, msg):# - frequency = gps callback freq. 
        """Callback to update states from EKF estimated state."""
        self.x, self.y, self.p, self.u, self.v, self.r = msg.data[:6]
              
        # to path fixed coordinate
        heron_pos = np.array([self.x, self.y])
        cte = (heron_pos - station_keeping_point)

        self.states = np.array([cte[0], cte[1], self.p, self.u, self.v, self.r, self.n1, self.n2])
        # print(self.p-wpt_psi)
        


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

        while not rospy.is_shutdown():
           
            self.state_estim, self.param_estim, self.param_filtered, L1_thruster, x_error = L1_control(self.states, self.state_estim, self.param_filtered, self.con_dt, self.param_estim, self.ii)

            # obtain mpc input
            self.n1 = L1_thruster[0]
            self.n2 = L1_thruster[1]

            print("state_estimation : ",self.state_estim)
            print("state_error", x_error)
            print("param_estimation : ",self.param_estim)
            print("Control input : ", L1_thruster)
            
            # Publish the control inputs (e.g., thrust commands)
            drive_msg = Drive()
            drive_msg.left = self.force_to_heron_command(self.n1,1)
            drive_msg.right = self.force_to_heron_command(self.n2,2)                        
            self.thrust_pub.publish(drive_msg)                                    
            
            self.ii = self.ii + 1

            
            self.rate.sleep()


if __name__ == '__main__':
    try:
        rospy.init_node('L1_LQR', anonymous=True)
        station_keeping = L1_LQR()
        rospy.loginfo("Starting L1_LQR control loop.")
        station_keeping.run()
    except rospy.ROSInterruptException:
        pass