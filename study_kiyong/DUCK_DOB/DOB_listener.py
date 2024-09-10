import rospy
from std_msgs.msg import Float64MultiArray
# from heron_msgs.msg import Drive
import math
import time
import numpy as np
from DOB import *


traj_xy = (353126.1653380032, 4026065.6140236068)
offset = np.array([353126.1653380032, 4026065.6140236068])
class HeronMPC:
    def __init__(self):
        # ROS settings
        self.rate = rospy.Rate(10)  # 10 Hz for ROS loop rate
        self.ekf_sub = rospy.Subscriber('/ekf/estimated_state', Float64MultiArray, self.ekf_callback)
        # self.ekf_sub = rospy.Subscriber('/cmd_drive', Drive, self.cmd_callback)
        # self.thrust_pub = rospy.Publisher('/cmd_drive', Drive)
        
        # Initial states and inputs
        self.x = self.y = self.p = self.u = self.v = self.r = 0.0
        self.n1 = self.n2 = 0.0
        self.states = np.zeros(8)
        self.MPC_thruster = np.zeros(2)

        self.param_estim = np.array([0.0, 0.0, 0.0, 0.0])
        self.param_filtered = np.array([0.0, 0.0, 0.0, 0.0])
        self.state_estim = np.array([0.0, 0.0,0.0, 0.0])
        
        
    def force_to_heron_command(self,force,left_right):
        """Convert force to Heron command, accounting for dead zones and offsets."""
        # left_right = 0 -> left, 1 -> right
        # dead zone -0.4~0.2
        # param : 25/0.6 (forward), 8/0.6 (backward)
        if force >= 0:
            n = force * 0.6/25
            n += 0.2
            if left_right == 1:
                n += 0.05 # offset left += 0.05
        else:
            n = force * 0.6/8         
            n -= 0.4
        return n
    
    def ekf_callback(self, msg):# - frequency = gps callback freq. 
        """Callback to update states from EKF estimated state."""
        self.x, self.y, self.p, self.u, self.v, self.r = msg.data[:6]
        self.states = np.array([self.x-offset[0], self.y-offset[1], self.p, self.u, self.v, self.r, self.n1, self.n2])


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
            t = time.time()
            ##### Reference States ######
  
            self.state_estim, self.param_estim = DOB(self.states, self.state_estim, 0.1, self.param_estim, self.MPC_thruster)
            M = 37.758
            print(self.param_estim*M)      
            # print(self.states[2])
            # print(yref[2])
            # print(self.n1, self.n2)
            # print(self.states)
            
            # Publish the control inputs (e.g., thrust commands)
            # drive_msg = Drive()
            # drive_msg.left = self.force_to_heron_command(self.n1,1)
            # drive_msg.right = self.force_to_heron_command(self.n2,2)                        
            # self.thrust_pub.publish(drive_msg)                                    
            
            
            self.rate.sleep()


if __name__ == '__main__':
    try:
        rospy.init_node('heron_mpc', anonymous=True)
        mpc = HeronMPC()
        rospy.loginfo("Starting MPC control loop.")
        mpc.run()
    except rospy.ROSInterruptException:
        pass
