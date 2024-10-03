import rospy
from std_msgs.msg import Float64MultiArray
from heron_msgs.msg import Drive
import math
import time
import numpy as np


traj_xy = (353152, 4026032)
offset = np.array([353152, 4026032])

class HeronKoopman:
    def __init__(self):
        # ROS settings
        self.rate = rospy.Rate(100)  # 10 Hz for ROS loop rate
        self.ekf_sub = rospy.Subscriber('/ekf/estimated_state', Float64MultiArray, self.ekf_callback, queue_size=100)
        self.thrust_pub = rospy.Publisher('/cmd_drive', Drive, queue_size=100)
        self.Koopman_pub = rospy.Publisher('/Koopman_data', Float64MultiArray, queue_size=100)  
        
        # Initial states and inputs
        self.x = self.y = self.p = self.u = self.v = self.r = 0.0
        self.n1 = self.n2 = 0.0
        self.states = np.zeros(8)
        self.state_on = 0
        self.n1_fix = self.n2_fix = 0.0

        # MPC parameter settings
        self.con_dt = 0.1 # control sampling time

        
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
        self.state_on = 1

    def random_with_dead_zone(self, low, high, dead_zone_low, dead_zone_high):
        # Generate a random number outside the dead zone
        if np.random.rand() < (dead_zone_low - low) / (high - low):
            return np.random.uniform(low, dead_zone_low)
        else:
            return np.random.uniform(dead_zone_high, high)

 
    def run(self):
        k = 0 # -> 현재 시간을 index로 표시 -> 그래야 ref trajectory설정가능(******** todo ********)

        while not rospy.is_shutdown():
            loop_start_time = time.time()
            t = time.time()
            max_input = 1
            min_input = -1
            change_limit = 0.1
            input_duration = 5
            ##### Reference States ######
            if self.state_on == 1:

                if k < 100:
                    if k % input_duration == 0:
                        n1_range = np.arange(max(self.n1 - change_limit, min_input), min(self.n1 + change_limit, max_input), 0.01)
                        n2_range = np.arange(max(self.n2 - change_limit, min_input), min(self.n2 + change_limit, max_input), 0.01)
                        self.n1 = np.random.choice(n1_range)
                        self.n1 = round(self.n1, 2)
                        self.n2 = np.random.choice(n2_range)
                        self.n2 = round(self.n2, 2)
                    
                    Koopman_data = [self.x-offset[0], self.y-offset[1], self.p, self.u, self.v, self.r, self.n1, self.n2]

                elif k < 150 and k>=100:    
                    Koopman_data = [0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0]
                    if k == 101:
                        self.n1_fix = self.random_with_dead_zone(-0.8, 0.8, -0.1, 0.1)
                        self.n2_fix = self.random_with_dead_zone(-0.8, 0.8, -0.1, 0.1)
                    
                    self.n1 = round(self.n1_fix,2)
                    self.n2 = round(self.n2_fix,2)
                elif k >= 150:
                    k = -1
                    self.n1 = round(self.n1_fix,2)
                    self.n2 = round(self.n2_fix,2)

                print(Koopman_data, k, self.n1, self.n2)

                drive_msg = Drive()
                drive_msg.left = self.n1
                drive_msg.right = self.n2
                self.thrust_pub.publish(drive_msg)                                    

                Koopman_msg = Float64MultiArray()
                Koopman_msg.data = Koopman_data
                self.Koopman_pub.publish(Koopman_msg)

                k = k+1
            
            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0, self.con_dt - elapsed_time)
            time.sleep(sleep_time)            



if __name__ == '__main__':
    try:
        rospy.init_node('heron_Koopman', anonymous=True)
        mpc = HeronKoopman()
        rospy.loginfo("Starting Data acquisition.")
        mpc.run()
    except rospy.ROSInterruptException:
        pass