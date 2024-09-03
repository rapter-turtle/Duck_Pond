#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from heron_msgs.msg import Drive
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Float64MultiArray
import numpy as np
import math
import time  # 시간 측정을 위해 추가

offset = np.array([353125.249598, 4026050.041195])

# Define the waypoints for the figure-eight trajectory
waypoints = np.array([
    [0.0, 10.0] + offset,
    [-10.0, 5.0] + offset,
    [0.0, 0.0] + offset,
    [10.0, -5.0] + offset,
    [0.0, -10.0] + offset,
    [0.0, 0.0] + offset
])

# Initial states and inputs
x = y = p = u = v = r = 0.0
n1 = n2 = 0.0
states = np.zeros(8)
con_dt = 0.1  # control sampling time

current_goal_index = 0
current_position = Pose()

# PID 제어를 위한 변수들
angle_integral = 0.0
distance_integral = 0.0
prev_angle_error = 0.0
prev_distance_error = 0.0

# PID 파라미터
Kp_angle = 1.5
Ki_angle = 0.0
Kd_angle = 0.2

Kp_distance = 0.5
Ki_distance = 0.0
Kd_distance = 0.1

def ekf_callback(msg):
    """Callback to update states from EKF estimated state."""
    global x, y, p, u, v, r, states
    x, y, p, u, v, r = msg.data[:6]
    states = np.array([x - offset[0], y - offset[1], p, u, v, r, n1, n2])
    #states = np.array([x + offset[0], y + offset[1], p, u, v, r, n1, n2])
    current_position.position.x = x
    current_position.position.y = y
    current_position.orientation.z = p
"""
def yaw_discontinuity(ref):
    # Handle yaw angle discontinuities.
    flag = [0.0] * 3
    flag[0] = abs(states[2] - ref)
    flag[1] = abs(states[2] - (ref - 2 * math.pi))
    flag[2] = abs(states[2] - (ref + 2 * math.pi))
    min_element_index = flag.index(min(flag))

    if min_element_index == 0:
        ref = ref
    elif min_element_index == 1:
        ref = ref - 2 * math.pi
    elif min_element_index == 2:
        ref = ref + 2 * math.pi
    return ref
"""
def pid_control(target, current, integral, prev_error, Kp, Ki, Kd):
    error = target - current
    integral += error
    derivative = error - prev_error
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral, error

def main():
    global current_goal_index, angle_integral, distance_integral, prev_angle_error, prev_distance_error

    rospy.init_node('position_based_navigation')
    pub_drive = rospy.Publisher('/cmd_drive', Drive, queue_size=10)
    pub_distance = rospy.Publisher('/distance_to_goal', Float64MultiArray, queue_size=10)  # 새로운 토픽
    rospy.Subscriber('/ekf/estimated_state', Float64MultiArray, ekf_callback)
    rate = rospy.Rate(10)  # 10Hz

    drive_msg = Drive()
    goal_tolerance = 1.0  # tolerance(m)

    while not rospy.is_shutdown():
        #start_time = time.time()  # 루프 시작 시간 기록

        goal_position = waypoints[current_goal_index]

        # distance between target point & current point
        distance_to_goal = math.sqrt((goal_position[0] - current_position.position.x)**2 + (goal_position[1] - current_position.position.y)**2)
        
        # 발행할 메시지 생성 및 발행
        distance_msg = Float64MultiArray()
        distance_msg.data = distance_to_goal
        pub_distance.publish(distance_msg)

        # calculating target angle
        goal_angle = math.atan2(goal_position[1] - current_position.position.y, goal_position[0] - current_position.position.x)

        # calculating current angle
        current_yaw = current_position.orientation.z  # EKF에서 가져온 yaw 값
        """ current_yaw = yaw_discontinuity(current_yaw) """

        # calculating angle between target & current angle
        angle_to_goal = goal_angle - current_yaw

        # angle normalizing into [-pi, pi]
        angle_to_goal = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi

        if distance_to_goal > goal_tolerance:
            # PID control
            angular_speed, angle_integral, prev_angle_error = pid_control(angle_to_goal, 0, angle_integral, prev_angle_error, Kp_angle, Ki_angle, Kd_angle)
            linear_speed, distance_integral, prev_distance_error = pid_control(0, -distance_to_goal, distance_integral, prev_distance_error, Kp_distance, Ki_distance, Kd_distance)

            # velocity limitation
            linear_speed = max(min(linear_speed, 0.8), -1.0)
            angular_speed = max(min(angular_speed, 0.5), -0.5)
        else:
            # move to next spot
            current_goal_index = (current_goal_index + 1) % len(waypoints)
            linear_speed = 0.0
            angular_speed = 0.0

        drive_msg.left = linear_speed - angular_speed + 0.05
        drive_msg.right = linear_speed + angular_speed

        pub_drive.publish(drive_msg)
        
        rate.sleep()

        print(distance_to_goal, angle_to_goal)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
