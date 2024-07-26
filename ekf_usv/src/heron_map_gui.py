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
from matplotlib.widgets import Button


x_actual_min = 352571.00
x_actual_max = 353531.94
y_actual_min = 4026778.56
y_actual_max = 4025815.16
x_width = x_actual_max-x_actual_min 

class SensorFusionEKF:
    def __init__(self):
        rospy.init_node('sensor_fusion_ekf', anonymous=True)
        self.time_data = []
        self.x_data = []
        self.y_data = []
        self.p_data = []
        self.u_data = []
        self.v_data = []
        self.r_data = []
        
        self.x_sensor_data = []
        self.y_sensor_data = []
        self.p_sensor_data = []
        self.u_sensor_data = []
        self.v_sensor_data = []
        self.r_sensor_data = []
        
        self.x_map = 0.0
        self.y_map = 0.0

        self.x_map_sensor = 0.0
        self.y_map_sensor = 0.0

        self.x = 0.0
        self.y = 0.0
        self.p = 0.0
        self.u = 0.0
        self.v = 0.0
        self.r = 0.0
        
        self.x_sensor = 0.0
        self.y_sensor = 0.0
        self.p_sensor = 0.0
        self.u_sensor = 0.0
        self.v_sensor = 0.0
        self.r_sensor = 0.0
        
        self.start_time = rospy.Time.now()
        self.current_time = rospy.Time.now()
        
        self.fig, self.ax = plt.subplots(2, 4, figsize=(12, 7))
        self.ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2, fig=self.fig)
        self.line1_m, = self.ax1.plot([], [], 'r-',label='measurement')    
        self.line1, = self.ax1.plot([], [], 'k-',label='ekf')    
        self.line1test, = self.ax1.plot([], [], 'm.',label='plot every 1-sec')    
        kaist_img = plt.imread("kaist.png")
        map_height, map_width = kaist_img.shape[:2]
        self.map_height = map_height
        self.map_width = map_width
        self.ax1.imshow(kaist_img[::-1],origin='lower')
        self.ax1.grid()
        self.ax1.set_xlabel('x position')
        self.ax1.set_ylabel('y position')
        self.ax1.set_title('Real-time Trajectory Plot')
        self.ax1.legend()

        self.ax2 = plt.subplot2grid((2, 4), (0, 2), fig=self.fig)
        self.line2_m, = self.ax2.plot([], [], 'r-', label='measurement', alpha=0.75)
        self.line2, = self.ax2.plot([], [], 'k-', label='ekf')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Surge (m/s)')

        self.ax3 = plt.subplot2grid((2, 4), (1, 2), fig=self.fig)
        self.line3_m, = self.ax3.plot([], [], 'r-', label='measurement', alpha=0.75)
        self.line3, = self.ax3.plot([], [], 'k-', label='ekf')
        self.ax3.set_xlabel('Time')
        self.ax3.set_ylabel('Sway (m/s)')

        self.ax4 = plt.subplot2grid((2, 4), (0, 3), fig=self.fig)
        self.line4_m, = self.ax4.plot([], [], 'r-', label='measurement', alpha=0.75)
        self.line4, = self.ax4.plot([], [], 'k-', label='ekf')
        self.ax4.set_xlabel('Time')
        self.ax4.set_ylabel('Yaw Rate (rad/s)')

        self.ax5 = plt.subplot2grid((2, 4), (1, 3), fig=self.fig)
        self.line5_m, = self.ax5.plot([], [], 'r-', label='measurement', alpha=0.75)
        self.line5, = self.ax5.plot([], [], 'k-', label='ekf')
        self.ax5.set_xlabel('Time')
        self.ax5.set_ylabel('Yaw (deg)')

        self.ax5.legend()
        self.ax4.legend()
        self.ax3.legend()
        self.ax2.legend()

        self.arrow_length = 0.5
        size = 1.5
        hullLength = 0.7 * size # Length of the hull
        hullWidth = 0.2 * size   # Width of each hull
        separation = 0.45 * size  # Distance between the two hulls
        bodyWidth = 0.25 * size  # Width of the body connecting the hulls

        # Define the vertices of the two hulls
        self.hull1 = np.array([[-hullLength/2, hullLength/2, hullLength/2, -hullLength/2, -hullLength/2, -hullLength/2],
                        [hullWidth/2, hullWidth/2, -hullWidth/2, -hullWidth/2, 0, hullWidth/2]])

        self.hull2 = np.array([[-hullLength/2, hullLength/2, hullLength/2, -hullLength/2, -hullLength/2, -hullLength/2],
                        [hullWidth/2, hullWidth/2, -hullWidth/2, -hullWidth/2, 0, hullWidth/2]])

        # Define the vertices of the body connecting the hulls
        self.body = np.array([[-bodyWidth/2, bodyWidth/2, bodyWidth/2, -bodyWidth/2, -bodyWidth/2],
                        [(separation-hullWidth)/2, (separation-hullWidth)/2, -(separation-hullWidth)/2, -(separation-hullWidth)/2, (separation-hullWidth)/2]])

        # Combine hulls into a single structure
        self.hull1[1, :] = self.hull1[1, :] + separation/2
        self.hull2[1, :] = self.hull2[1, :] - separation/2              
        
        # Rotation matrix for the heading
        R = np.array([[np.cos(self.p), -np.sin(self.p)],
                    [np.sin(self.p), np.cos(self.p)]])
        # Rotate the hulls and body
        hull1_R = R @ self.hull1
        hull2_R = R @ self.hull2
        body_R = R @ self.body
        # Translate the hulls and body to the specified position
        hull1_R += np.array([0,0]).reshape(2, 1)
        hull2_R += np.array([0,0]).reshape(2, 1)
        body_R += np.array([0,0]).reshape(2, 1)
        direction = np.array([np.cos(self.p), np.sin(self.p)]) * self.arrow_length
        # Plot the ASV
        self.heron_p1 = self.ax1.fill(hull1_R[0, :], hull1_R[1, :], 'b', alpha=0.35)
        self.heron_p2 = self.ax1.fill(hull2_R[0, :], hull2_R[1, :], 'b', alpha=0.35)
        self.heron_p3 = self.ax1.fill(body_R[0, :], body_R[1, :], 'b', alpha=0.35)
        self.heron_p4 = self.ax1.arrow(0,0, direction[0], direction[1], head_width=0.1, head_length=0.1, fc='b', ec='b')

        self.axins = self.ax1.inset_axes([0.65, 0.01, 0.34, 0.34])  # position and size of inset axis

        self.fig.tight_layout()  # axes 사이 간격을 적당히 벌려줍니다.
        self.ekf_sub = rospy.Subscriber('/ekf/estimated_state', Float64MultiArray, self.ekf_callback)

    
        reset_ax = plt.axes([0.41, 0.05, 0.05, 0.03])
        self.reset_button = Button(reset_ax, 'Reset')
        self.reset_button.on_clicked(self.reset_plots)
        

    def ekf_callback(self, msg):# - 주기가 gps callback 주기랑 같음 - gps data callback받으면 ekf에서 publish 하기때문
        self.x = msg.data[0]
        self.y = msg.data[1]
        self.p = msg.data[2]
        self.u = msg.data[3]
        self.v = msg.data[4]
        self.r = msg.data[5]
        
        self.x_sensor = msg.data[6]
        self.y_sensor = msg.data[7]
        self.p_sensor = msg.data[8]
        self.u_sensor = msg.data[9]
        self.v_sensor = msg.data[10]
        self.r_sensor = msg.data[11]
        
        self.current_time = (rospy.Time.now() - self.start_time).to_sec()
            
        self.x_map = (self.x - x_actual_min)/(x_actual_max - x_actual_min)*self.map_width
        self.y_map = self.map_height-(self.y - y_actual_min)/(y_actual_max - y_actual_min)*self.map_height
        
        self.x_map_sensor = (self.x_sensor - x_actual_min)/(x_actual_max - x_actual_min)*self.map_width
        self.y_map_sensor = self.map_height-(self.y_sensor - y_actual_min)/(y_actual_max - y_actual_min)*self.map_height
        # print(x_map,y_map)
        
        self.x_data.append(self.x_map)
        self.y_data.append(self.y_map)        
        self.p_data.append(self.p*180/np.pi)       
        self.u_data.append(self.u)
        self.v_data.append(self.v)
        self.r_data.append(self.r)

        self.x_sensor_data.append(self.x_map_sensor)
        self.y_sensor_data.append(self.y_map_sensor)        
        self.p_sensor_data.append(self.p_sensor*180/np.pi)       
        self.u_sensor_data.append(self.u_sensor)
        self.v_sensor_data.append(self.v_sensor)
        self.r_sensor_data.append(self.r_sensor)

        self.time_data.append(self.current_time)

    
    def update_plot(self, frame):
        if len(self.time_data)>1:

            self.line1_m.set_data(self.x_sensor_data, self.y_sensor_data)
            self.line1.set_data(self.x_data, self.y_data)
            self.line1test.set_data(self.x_data[::20], self.y_data[::20])
            self.ax1.set_xlim(520, 570)
            self.ax1.set_ylim(900, 960)
            
            # Rotation matrix for the heading
            R = np.array([[np.cos(self.p), -np.sin(self.p)],
                        [np.sin(self.p), np.cos(self.p)]])
            # Rotate the hulls and body
            hull1_R = R @ self.hull1
            hull2_R = R @ self.hull2
            body_R = R @ self.body
            # Translate the hulls and body to the specified position
            hull1_R += np.array([self.x_map, self.y_map]).reshape(2, 1)
            hull2_R += np.array([self.x_map, self.y_map]).reshape(2, 1)
            body_R += np.array([self.x_map, self.y_map]).reshape(2, 1)
            direction = np.array([np.cos(self.p), np.sin(self.p)]) * self.arrow_length
            # Plot the ASV
            self.heron_p1[0].set_xy(np.column_stack((hull1_R[0, :], hull1_R[1, :])))
            self.heron_p2[0].set_xy(np.column_stack((hull2_R[0, :], hull2_R[1, :])))
            self.heron_p3[0].set_xy(np.column_stack((body_R[0, :], body_R[1, :])))

            self.heron_p4.remove()
            self.heron_p4 = self.ax1.arrow(self.x_map, self.y_map, direction[0], direction[1], head_width=0.1, head_length=0.1, fc='g', ec='g')                  

            self.line2_m.set_data(self.time_data, self.u_sensor_data)
            self.line2.set_data(self.time_data, self.u_data)
            self.ax2.set_xlim(self.time_data[-1]-20, self.time_data[-1])
            self.ax2.set_ylim(-1, 2)
            
            self.line3_m.set_data(self.time_data, self.v_sensor_data)
            self.line3.set_data(self.time_data, self.v_data)
            self.ax3.set_xlim(self.time_data[-1]-20, self.time_data[-1])
            self.ax3.set_ylim(-2, 2)

            self.line4_m.set_data(self.time_data, self.r_sensor_data)
            self.line4.set_data(self.time_data, self.r_data)
            self.ax4.set_xlim(self.time_data[-1]-20, self.time_data[-1])
            self.ax4.set_ylim(-1.5, 1.5)

            self.line5_m.set_data(self.time_data, self.p_sensor_data)
            self.line5.set_data(self.time_data, self.p_data)
            self.ax5.set_xlim(self.time_data[-1]-20, self.time_data[-1])
            
            # self.ax2.set_xlim(self.time_data[0], self.time_data[-1])
            # self.ax3.set_xlim(self.time_data[0], self.time_data[-1])
            # self.ax4.set_xlim(self.time_data[0], self.time_data[-1])
            # self.ax5.set_xlim(self.time_data[0], self.time_data[-1])

            self.ax1.relim()
            self.ax2.relim()
            self.ax3.relim()
            self.ax4.relim()
            self.ax5.relim()

            self.ax1.autoscale_view()
            self.ax2.autoscale_view()
            self.ax3.autoscale_view()
            self.ax4.autoscale_view()
            self.ax5.autoscale_view()
            
            
            
            # Update the inset plot
            self.axins.clear()
            self.axins.plot(self.x_data, self.y_data, 'k-')
            self.axins.plot(self.x_sensor_data, self.y_sensor_data, 'r-')
            self.axins.fill(hull1_R[0, :], hull1_R[1, :], 'b', alpha=0.35)
            self.axins.fill(hull2_R[0, :], hull2_R[1, :], 'b', alpha=0.35)
            self.axins.fill(body_R[0, :], body_R[1, :], 'b', alpha=0.35)
            self.axins.arrow(self.x_map, self.y_map, direction[0], direction[1], head_width=0.1, head_length=0.1, fc='g', ec='g')
            self.axins.axis('equal')
            self.axins.set_xlim(self.x_map-3, self.x_map+3)
            self.axins.set_ylim(self.y_map-3, self.y_map+3)
            self.axins.get_xaxis().set_visible(False)
            self.axins.get_yaxis().set_visible(False)
            
            
            self.fig.tight_layout()  # axes 사이 간격을 적당히 벌려줍니다.

            return self.line1, self.line2, self.line3, self.line4, self.line5

    def reset_plots(self, event):
        self.start_time = rospy.Time.now()
        self.current_time = rospy.Time.now()
        
        self.time_data = []
        self.x_data = []
        self.y_data = []
        self.p_data = []
        self.u_data = []
        self.v_data = []
        self.r_data = []
        
        self.x_sensor_data = []
        self.y_sensor_data = []
        self.p_sensor_data = []
        self.u_sensor_data = []
        self.v_sensor_data = []
        self.r_sensor_data = []

        self.line1_m.set_data([], [])
        self.line1.set_data([], [])
        self.line1test.set_data([], [])
        self.line2_m.set_data([], [])
        self.line2.set_data([], [])
        self.line3_m.set_data([], [])
        self.line3.set_data([], [])
        self.line4_m.set_data([], [])
        self.line4.set_data([], [])
        self.line5_m.set_data([], [])
        self.line5.set_data([], [])

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.ax4.relim()
        self.ax4.autoscale_view()
        self.ax5.relim()
        self.ax5.autoscale_view()
        
        
    def run(self):
        ani = FuncAnimation(self.fig, self.update_plot, blit=False, interval=100)
        plt.show()

if __name__ == '__main__':
    ekf = SensorFusionEKF()
    try:
        ekf.run()
    except rospy.ROSInterruptException:
        pass