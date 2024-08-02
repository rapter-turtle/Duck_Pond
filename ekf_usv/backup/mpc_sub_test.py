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
from heron_msgs.msg import Drive
import pandas as pd
import time
from mpc_msgs.msg import MPCTraj, MPC_State

x_actual_min = 352571.00 - 2
x_actual_max = 353531.94 - 2
y_actual_min = 4026778.56 - 4
y_actual_max = 4025815.16 - 4
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
        self.thrust_left_data = []
        self.thrust_right_data = []
        
        self.thrust_left = []
        self.thrust_right = []
        
        self.x_sensor_data = []
        self.y_sensor_data = []
        self.p_sensor_data = []
        self.u_sensor_data = []
        self.v_sensor_data = []
        self.r_sensor_data = []
        
        self.x_map = []
        self.y_map = []

        self.x_map_sensor = []
        self.y_map_sensor = []

        self.x = []
        self.y = []
        self.p = []
        self.u = []
        self.v = []
        self.r = []
        
        self.x_sensor = []
        self.y_sensor = []
        self.p_sensor = []
        self.u_sensor = []
        self.v_sensor = []
        self.r_sensor = []
        
        
        self.pred_x = []
        self.pred_y = []
        self.ref_x = []
        self.ref_y = []
        
        self.current_time = []
        
        self.fig, self.ax = plt.subplots(3, 4, figsize=(12, 7))
        self.ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=4, rowspan=3, fig=self.fig)
        self.line1_m, = self.ax1.plot([], [], 'r-',label='measurement')    
        self.line1, = self.ax1.plot([], [], 'k-',label='ekf')    
        
        self.pred_line, = self.ax1.plot([], [], 'm-', label='Predicted')
        self.ref_line, = self.ax1.plot([], [], 'b--', label='Reference')
        
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

        self.fig.tight_layout()  # axes 사이 간격을 적당히 벌려줍니다.
        self.mpc_sub = rospy.Subscriber('/mpc_vis', MPCTraj, self.mpc_callback)

        
    def mpc_callback(self, msg):# - 주기가 gps callback 주기랑 같음 - gps data callback받으면 ekf에서 publish 하기때문        
        # Clear previous data
        self.pred_x.clear()
        self.pred_y.clear()
        self.ref_x.clear()
        self.ref_y.clear()

        # Extract predicted and reference trajectories
        for state in msg.state:
            self.pred_x.append(state.x)
            self.pred_y.append(state.y)

        for ref in msg.ref:
            self.ref_x.append(ref.x)
            self.ref_y.append(ref.y)
            
        print(self.ref_x)
        
    def update_plot(self, frame):
        self.pred_line.set_data(self.pred_x,self.pred_y)
        self.ref_line.set_data(self.ref_x,self.ref_y)

        self.ax1.set_xlim(-10, 10)
        self.ax1.set_ylim(-10, 10)
    
        self.ax1.relim()
        self.ax1.autoscale_view()
            
        return self.line1
        
    def run(self):
        ani = FuncAnimation(self.fig, self.update_plot, blit=False, interval=100)
        plt.show()

if __name__ == '__main__':
    ekf = SensorFusionEKF()
    try:
        ekf.run()
    except rospy.ROSInterruptException:
        pass