#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import tf
import math

class SensorFusionEKF:
    def __init__(self):
        rospy.init_node('sensor_fusion_ekf', anonymous=True)
        self.ekf_sub = rospy.Subscriber('/ekf/estimated_state', Float64MultiArray, self.ekf_callback)

        self.time_data = []

        self.gps_x_data = []
        self.gps_y_data = []
        self.imu_psi_data = []
        self.imu_r_data = []
        self.imu_ax_data = []
        self.imu_ay_data = []

        self.x_data = []
        self.y_data = []
        self.yaw_data = []
        self.surge_data = []
        self.sway_data = []
        self.yaw_rate_data = []
        
        self.callback_count = 0
        # Set up plot
        self.fig, self.ax = plt.subplots(2, 4, figsize=(15, 15))
        
        # Use the left 3x2 space for the trajectory plot
        self.ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2, fig=self.fig)
        self.line1, = self.ax1.plot([], [], 'k-',label='ekf')    
        self.line1gps, = self.ax1.plot([], [], 'r-',label='gps')    
        # self.ax1.set_xlim(-600, -400)
        # self.ax1.set_ylim(100, 300)
        self.ax1.set_xlabel('X Position')
        self.ax1.set_ylabel('Y Position')
        self.ax1.set_title('Real-time Trajectory Plot')

        # Bottom-right 1x1 subplot for Surge
        self.ax2 = plt.subplot2grid((2, 4), (0, 2), fig=self.fig)
        self.line2, = self.ax2.plot([], [], 'k-', label='ekf')
        # self.ax2.set_xlim(0, 100)
        # self.ax2.set_ylim(-2, 2)
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Surge (m/s)')

        # Bottom-middle 1x1 subplot for Sway
        self.ax3 = plt.subplot2grid((2, 4), (1, 2), fig=self.fig)
        self.line3, = self.ax3.plot([], [], 'k-', label='ekf')
        # self.ax3.set_xlim(0, 100)
        # self.ax3.set_ylim(-1, 1)
        self.ax3.set_xlabel('Time')
        self.ax3.set_ylabel('Sway (m/s)')

        # Bottom-right 1x1 subplot for Yaw Rate
        self.ax4 = plt.subplot2grid((2, 4), (0, 3), fig=self.fig)
        self.line4, = self.ax4.plot([], [], 'k-', label='Yaw_Rate-ekf')
        self.line4imu, = self.ax4.plot([], [], 'r-', label='Yaw_Rate-imu')
        self.ax4.set_xlabel('Time')
        self.ax4.set_ylabel('Yaw Rate (rad/s)')

        self.ax5 = plt.subplot2grid((2, 4), (1, 3), fig=self.fig)
        self.line5, = self.ax4.plot([], [], 'k-', label='Yaw-ekf')
        self.line5imu, = self.ax4.plot([], [], 'r-', label='Yaw-imu')
        self.ax5.set_xlabel('Time')
        self.ax5.set_ylabel('Yaw (deg)')

        self.start_time = rospy.Time.now()
        self.current_time = rospy.Time.now()
        
        

    def ekf_callback(self, msg):# - 주기가 gps callback 주기랑 같음 - gps data callback받으면 ekf에서 publish 하기때문
        self.ekf_x = msg.data[0]
        self.ekf_y = msg.data[1]
        self.ekf_psi = msg.data[2]
        self.ekf_surge = msg.data[3]
        self.ekf_sway = msg.data[4]
        self.ekf_r = msg.data[5]

        self.gps_x = msg.data[6]
        self.gps_y = msg.data[7]

        self.imu_psi = msg.data[8]
        self.imu_r = msg.data[9]
        self.imu_ax = msg.data[10]
        self.imu_ay = msg.data[11]

        # Append data to lists
        self.current_time = (rospy.Time.now() - self.start_time).to_sec()
        self.time_data.append(self.current_time)
        
        self.gps_x_data.append(self.gps_x)
        self.gps_y_data.append(self.gps_y)
        self.imu_psi_data.append(self.imu_psi)
        self.imu_r_data.append(self.imu_r)

        self.x_data.append(self.ekf_x)
        self.y_data.append(self.ekf_y)
        self.yaw_data.append(self.ekf_psi)
        self.yaw_rate_data.append(self.ekf_r)
        self.surge_data.append(self.ekf_surge)                
        self.sway_data.append(self.ekf_sway)               

        # rospy.loginfo("Position: (%.2f, %.2f), Yaw: %.2f", position.x, position.y, yaw*180/3.141592)
        # rospy.loginfo("Velocities: (%.2f, %.2f, %.2f)", twist.linear.x, twist.linear.y, twist.angular.z*180/3.141592)

    def update_plot(self, frame):
        # Update trajectory plot

        # self.line1.set_data(self.x_data, self.y_data)
        self.line1.set_data(self.x_data, self.y_data)
        self.line1gps.set_data(self.gps_x_data, self.gps_y_data)
        # self.ax1.set_xlim(self.x_data[-1]-25, self.x_data[-1]+25)
        # self.ax1.set_ylim(self.y_data[-1]-25, self.y_data[-1]+25)

        # Update surge and sway velocity plot

        self.line2.set_data(self.time_data, self.surge_data)
        self.ax2.set_xlim(self.time_data[-1]-10, self.time_data[-1])

        self.line3.set_data(self.time_data, self.sway_data)
        self.ax3.set_xlim(self.time_data[-1]-10, self.time_data[-1])

        # self.line4.set_data(self.time_data, self.yaw_rate_data)
        # self.line4imu.set_data(self.time_data, self.imu_r_data)
        self.ax4.set_xlim(self.time_data[-1]-10, self.time_data[-1])

        self.line5.set_data(self.time_data, self.yaw_data)
        self.line5imu.set_data(self.time_data, self.imu_psi_data)

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.ax4.relim()
        self.ax4.autoscale_view()
        return self.line1, self.line2, self.line3, self.line4


    def run(self):
        ani = FuncAnimation(self.fig, self.update_plot, blit=False, interval=100)
        plt.show()

if __name__ == '__main__':
    ekf = SensorFusionEKF()
    try:
        ekf.run()
    except rospy.ROSInterruptException:
        pass