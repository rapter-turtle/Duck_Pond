import rospy
from std_msgs.msg import Float64MultiArray
from heron_msgs.msg import Drive
from sensor_msgs.msg import Imu  # Import the message type for IMU
import math
import time
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from DOB import *

traj_xy = (353126.1653380032, 4026065.6140236068)
offset = np.array([353126.1653380032, 4026065.6140236068])

class HeronMPC:
    def __init__(self):
        # ROS settings
        self.rate = rospy.Rate(10)  # 10 Hz for ROS loop rate
        self.last_ekf_time = 0.0
        # Timestamp from IMU for regulating the callback rate
        self.last_imu_time = None
        self.current_imu_time = None
        self.start_time = time.time()  # Record the start time for the rolling window

        self.ekf_sub = rospy.Subscriber('/ekf/estimated_state', Float64MultiArray, self.ekf_callback)
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)  # Subscribe to IMU data
        self.cmd_sub = rospy.Subscriber('/cmd_drive', Drive, self.cmd_callback)  # Subscribe to IMU data
        
        # Initial states and inputs
        self.x = self.y = self.p = self.u = self.v = self.r = 0.0
        self.n1 = self.n2 = 0.0
        self.states = np.zeros(8)
        self.MPC_thruster = np.zeros(2)

        self.param_estim = np.array([0.0, 0.0, 0.0, 0.0])
        self.param_filtered = np.array([0.0, 0.0, 0.0, 0.0])
        self.state_estim = np.array([0.0, 0.0, 0.0, 0.0])
        self.n1 = 0.0
        self.n2 = 0.0

        # Set up live plotting
        plt.ion()  # Interactive mode on
        self.fig, self.ax = plt.subplots()
        self.time_history = []  # To track time for the x-axis
        self.param_estim_history = [[], [],[], []]  # Store param_estim[2] and param_estim[3] history for plotting
        
        # Create a plot line for param_estim[2] and param_estim[3], and add labels for legend
        self.lines = [
            self.ax.plot([], [], label="X Disturbance")[0],
            self.ax.plot([], [], label="Y Disturbance")[0],
            self.ax.plot([], [], label="X thrust")[0],
            self.ax.plot([], [], label="Y thrust")[0]
        ]
        
        self.ax.set_xlim(0, 10)  # Set x-axis limits to 10 seconds for the rolling window
        self.ax.set_ylim(-60, 60)
        self.ax.set_title("Headpoint Disturbance Estimation")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Disturbance(N)")
        self.ax.grid()
        self.ax.legend()  # Add the legend to the plot

    def imu_callback(self, msg):
        """Callback for IMU data to extract time information."""
        self.current_imu_time = msg.header.stamp.to_sec()  # Get timestamp in seconds from IMU message
        print(self.current_imu_time)

    def cmd_callback(self, msg):
        """Callback for IMU data to extract time information."""
        # self.current_imu_time = msg.header.stamp.to_sec()  # Get timestamp in seconds from IMU message
        self.n1 = msg.left
        self.n2 = msg.right
        
        # print(msg.left)

    def force_to_heron_command(self, force, left_right):
        """Convert force to Heron command, accounting for dead zones and offsets."""
        if force >= 0:
            n = force * 0.6 / 25
            n += 0.2
            if left_right == 1:
                n += 0.05  # offset left += 0.05
        else:
            n = force * 0.6 / 8
            n -= 0.4
        return n

    def ekf_callback(self, msg):
        """Callback to update states from EKF estimated state, regulated using IMU time."""
        if self.current_imu_time is None:
            return  # Wait until we receive IMU data with timestamp

        if self.last_imu_time is None or self.current_imu_time - self.last_imu_time >= 0.1:  # Check if 0.1 seconds have passed (10 Hz)
            self.x, self.y, self.p, self.u, self.v, self.r = msg.data[:6]
            self.states = np.array([self.x - offset[0], self.y - offset[1], self.p, self.u, self.v, self.r, self.n1, self.n2])
            self.last_imu_time = self.current_imu_time  # Update the last processed time
            print(self.current_imu_time)

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

    def update_plot(self):
        """Update the live plot with the latest parameter estimations (param_estim[2] and param_estim[3])."""
        current_time = time.time() - self.start_time  # Get the elapsed time in seconds
        M = 40
        self.time_history.append(current_time)  # Track 40 for the x-axis
        self.param_estim_history[0].append(-self.param_filtered[2] * M)  # Store history for param_estim[2]
        self.param_estim_history[1].append(self.param_filtered[3] * M)  # Store history for param_estim[3]
        self.param_estim_history[2].append((self.n1 + self.n2)*M*np.cos(self.p) - 0.3*(-self.n1 + self.n2)*M*np.sin(self.p))  # Store history for param_estim[2]
        self.param_estim_history[3].append((self.n1 + self.n2)*M*np.sin(self.p) + 0.3*(-self.n1 + self.n2)*M*np.cos(self.p))  # Store history for param_estim[3]

        # Keep only the last 10 seconds of data
        while self.time_history and self.time_history[-1] - self.time_history[0] > 10:
            self.time_history.pop(0)
            self.param_estim_history[0].pop(0)
            self.param_estim_history[1].pop(0)
            self.param_estim_history[2].pop(0)
            self.param_estim_history[3].pop(0)

        # Update the line data for param_estim[2] and param_estim[3]
        for i in range(4):
            self.lines[i].set_xdata(self.time_history)
            self.lines[i].set_ydata(self.param_estim_history[i])

        self.ax.set_xlim(max(0, self.time_history[0]), self.time_history[-1])  # Adjust x-axis dynamically
        self.ax.relim()  # Recalculate limits
        self.ax.autoscale_view()  # Autoscale the view to fit data
        self.fig.canvas.draw()  # Redraw the plot
        self.fig.canvas.flush_events()  # Flush any pending GUI events

    def run(self):
        while not rospy.is_shutdown():
            ##### Reference States ######
            self.state_estim, self.param_estim, self.param_filtered = DOB(self.states, self.state_estim, 0.1, self.param_estim, self.MPC_thruster, self.param_filtered)

            # Update the live plot every iteration
            self.update_plot()

            self.rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('heron_mpc', anonymous=True)
        mpc = HeronMPC()
        rospy.loginfo("Starting MPC control loop.")
        mpc.run()
    except rospy.ROSInterruptException:
        pass
