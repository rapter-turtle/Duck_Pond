#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float32
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import tf
import math

class ThrustCommandPublisher:
    def __init__(self):
        rospy.init_node('thrust_command_publisher', anonymous=True)
        self.lcmd_pub = rospy.Publisher('/wamv/thrusters/left_thrust_cmd', Float32, queue_size=1)
        self.rcmd_pub = rospy.Publisher('/wamv/thrusters/right_thrust_cmd', Float32, queue_size=1)

        self.rate = rospy.Rate(10)

        # Initialize the thrust commands
        self.left_thrust_cmd = Float32()
        self.right_thrust_cmd = Float32()


    def publish_thrust_commands(self):
        while not rospy.is_shutdown():
            # Set the desired thrust commands (example values)
            self.left_thrust_cmd.data = 1.0  # Example value for left thruster
            self.right_thrust_cmd.data = 1.0  # Example value for right thruster
            
            # Publish the thrust commands
            self.lcmd_pub.publish(self.left_thrust_cmd)
            self.rcmd_pub.publish(self.right_thrust_cmd)
            
            # Sleep to maintain the desired rate
            self.rate.sleep()

if __name__ == '__main__':
    try:
        thrust_publisher = ThrustCommandPublisher()
        thrust_publisher.publish_thrust_commands()
    except rospy.ROSInterruptException:
        pass