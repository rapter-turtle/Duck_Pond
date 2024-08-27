import rospy
from std_msgs.msg import Float64MultiArray
from heron_msgs.msg import Drive
from mpc_msgs.msg import MPCTraj, MPC_State, Obs_State
import math
import time
import numpy as np
import sys

class HeronMPC:
    def __init__(self):
        # ROS settings
        self.rate = rospy.Rate(10)  # 10 Hz for ROS loop rate
        self.ekf_sub = rospy.Subscriber('/ekf/estimated_state', Float64MultiArray, self.ekf_callback)
        self.thrust_pub = rospy.Publisher('/cmd_drive', Drive)
        
        # Initial states and inputs
        self.x = self.y = self.p = self.u = self.v = self.r = 0.0
        self.n1 = self.n2 = 0.0
        self.states = np.zeros(8)
        self.con_dt = 0.1 # control sampling tim
        
        self.receive_gps = 0
        self.init_p = 0.0
        
    def force_to_heron_command(self,force,left_right):
        """Convert force to Heron command, accounting for dead zones and offsets."""
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
        if self.receive_gps == 0:
            self.init_p = self.p
            self.receive_gps = 1


    def run(self,args):
        del_heading = int(args[1])
        del_input_max = int(args[2])
        thrust = int(args[3])
        print('del heading [deg] :', del_heading)
        print('del input :', del_input_max)
        print('Thrust :', thrust)
        
        del_input = 0.5
        self.n1 = thrust
        self.n2 = thrust
        flag = 1
        k = 0
        while not rospy.is_shutdown():
            if self.receive_gps:
                k += 1
                # Publish the control inputs (e.g., thrust commands)
                drive_msg = Drive()                
                if k < 50:           
                    print(k)
                    print('time delay') 
                else:                        
                    self.n1 -= del_input
                    self.n2 += del_input
                    self.n1 = np.clip(self.n1,thrust-del_input_max,thrust+del_input_max)
                    self.n2 = np.clip(self.n2,thrust-del_input_max,thrust+del_input_max)
                    
                self.n1 = np.clip(self.n1,-8,25)
                self.n2 = np.clip(self.n2,-8,25)

                drive_msg.left = self.force_to_heron_command(self.n1,1)
                drive_msg.right = self.force_to_heron_command(self.n2,2)                        
                print(self.init_p, self.p)
                print(drive_msg.left, drive_msg.right)
                self.thrust_pub.publish(drive_msg)                                    
                self.rate.sleep()

                if flag*(self.p - self.init_p) > del_heading*np.pi/180:
                    del_input = del_input * -1
                    flag = flag * -1


if __name__ == '__main__':
    try:
        rospy.init_node('heron_zigzag', anonymous=True)
        mpc = HeronMPC()
        rospy.loginfo("Starting zigzag control loop.")
        mpc.run(sys.argv)
    except rospy.ROSInterruptException:
        pass