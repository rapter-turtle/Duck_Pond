#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from heron_msgs.msg import Drive

class DataPlotter:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('data_plotter', anonymous=True)
        
        # Initialize lists for data storage
        self.xdata = []
        self.ydata1 = []
        self.ydata2 = []
        self.n1 = 0.0
        self.n2 = 0.0
        # Set up the matplotlib figure and axes
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Real-time Plot of /l1_data')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Data Values')
        self.lines1, = self.ax.plot([], [], label='3rd Data Element')
        self.lines2, = self.ax.plot([], [], label='4th Data Element')
        self.ax.legend()

        # Timer variable for incrementing time
        self.current_time = 0.0

        # Subscribe to the ROS topic
        self.data_sub = rospy.Subscriber('/l1_data', Float64MultiArray, self.data_callback)
        self.thrust_sub = rospy.Subscriber('/cmd_drive', Drive, self.thrust_callback)

        # Use FuncAnimation to update the plot
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)

    def data_callback(self, msg):
        # Handle incoming messages
        if len(msg.data) >= 4:
            data3 = msg.data[6]  # Third data element
            data4 = msg.data[7]  # Fourth data element

            # Append new data to the lists
            self.xdata.append(self.current_time)
            self.ydata1.append(data3)
            self.ydata2.append(data4)
            self.current_time += 0.1  # Increment time by 0.1 seconds

    def thrust_callback(self, msg):# - frequency = gps callback freq. 
              
        self.n1 = msg.left
        self.n2 = msg.right

    def update_plot(self, frame):
        # Only update if there is data
        if self.xdata:
            self.lines1.set_xdata(self.xdata)
            self.lines1.set_ydata(self.ydata1)
            self.lines2.set_xdata(self.xdata)
            self.lines2.set_ydata(self.ydata2)

            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()

    def run(self):
        plt.show()
        rospy.spin()  # Keep the program running

if __name__ == '__main__':
    plotter = DataPlotter()
    rospy.loginfo("Starting the data plotter node.")
    plotter.run()
