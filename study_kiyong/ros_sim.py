import rospy
from std_msgs.msg import Float64MultiArray
from heron_msgs.msg import Drive
from sensor_msgs.msg import Imu
import numpy as np
import tf.transformations as tf

class ShipSimulator:
    def __init__(self):
        # ROS settings
        self.rate = rospy.Rate(10)  # 10 Hz
        self.state_pub = rospy.Publisher('/ekf/estimated_state', Float64MultiArray, queue_size=10)
        self.imu_pub = rospy.Publisher('/imu/data', Imu, queue_size=10)
        self.control_sub = rospy.Subscriber('/cmd_drive', Drive, self.control_callback)
        station_keeping_point = np.array([353148, 4026039])
        station_keeping_point = np.array([353148, 4026019])
        # Initialize state [x, y, psi, u, v, r]
        self.ship_state = np.zeros(6)
        self.ship_state[0] = station_keeping_point[0]
        self.ship_state[1] = station_keeping_point[1]
        self.ship_state[2] = 3.141592*0.5
        self.control_input = np.zeros(2)  # [n1, n2]
        self.dt = 0.1  # Time step for the simulation
    
    def control_callback(self, msg):
        """Callback to update control input from the control node."""
        self.control_input[0] = self.command_to_force(msg.right, 1)
        self.control_input[1] = self.command_to_force(msg.left, 2)

    def command_to_force(self, command, left_right):
        """Convert Heron command back to force."""
        if command >= 0.2:
            force = (command - 0.2) * 25 / 0.6
            if left_right == 1:
                force -= 0.1 * 25 / 0.6  # offset adjustment for left side
            if force > 25:
                force = 25
        else:
            force = (command + 0.4) * 8 / 0.6
            if force < -2:
                force = -2
        return force

    def recover_simulator(self, ship, control_input, dt):
        """Simulate the ship dynamics given the current state and control input."""
        M = 37.758  # Mass [kg]
        I = 18.35   # Inertial tensor [kg m^2]
        Xu = 8.9149
        Xuu = 11.2101
        Nr = 16.9542
        Nrrr = 12.8966
        Yv = 15
        Yvv = 3
        Yr = 6
        Nv = 6
        dist = 0.3  # 30cm

        # Extract states and controls
        psi, u, v, r = ship[2], ship[3], ship[4], ship[5]
        n1, n2 = control_input[0], control_input[1]
        eps = 1e-5

        disturbance = np.array([0.0,0.0])
        disturbance_uv = np.array([disturbance[0]*np.cos(-psi) - disturbance[1]*np.sin(-psi),disturbance[0]*np.sin(-psi) + disturbance[1]*np.cos(-psi)])
        
        # Dynamics calculation
        xdot = np.array([
            u * np.cos(psi) - v * np.sin(psi),
            u * np.sin(psi) + v * np.cos(psi),
            r,
            ((n1 + n2) + disturbance_uv[0] - (Xu + Xuu * np.sqrt(u * u + eps)) * u) / M,
            ( disturbance_uv[1] -Yv * v - Yvv * np.sqrt(v * v + eps) * v - Yr * r) / M,
            ((-n1 + n2) * dist - (Nr + Nrrr * r * r) * r - Nv * v) / I
        ])

        # Update the ship state
        ship = xdot * dt + ship
        ship[2] = (ship[2] + np.pi) % (2 * np.pi) - np.pi  # Normalize psi between [-pi, pi]

        return ship

    def publish_imu(self):
        """Publish simulated IMU data."""
        imu_msg = Imu()
        imu_msg.header.stamp = rospy.Time.now()
        imu_msg.header.frame_id = "base_link"

        # Simulate the orientation from yaw (psi)
        quaternion = tf.quaternion_from_euler(0, 0, self.ship_state[2])
        imu_msg.orientation.x = quaternion[0]
        imu_msg.orientation.y = quaternion[1]
        imu_msg.orientation.z = quaternion[2]
        imu_msg.orientation.w = quaternion[3]

        # Simulate angular velocity (r) and linear acceleration (u, v)
        imu_msg.angular_velocity.z = self.ship_state[5]  # r
        imu_msg.linear_acceleration.x = self.ship_state[3]  # u
        imu_msg.linear_acceleration.y = self.ship_state[4]  # v

        self.imu_pub.publish(imu_msg)

    def run(self):
        while not rospy.is_shutdown():
            # Update ship state based on the control input
            self.ship_state = self.recover_simulator(self.ship_state, self.control_input, self.dt)
            
            # Publish the updated state
            state_msg = Float64MultiArray()
            # state_msg.data = [self.ship_state, self.ship_state]
            state_msg.data = np.concatenate((self.ship_state, self.ship_state)).tolist()

            self.state_pub.publish(state_msg)
            
            # Publish the simulated IMU data
            self.publish_imu()
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('ship_simulator', anonymous=True)
        simulator = ShipSimulator()
        rospy.loginfo("Starting ship simulator node.")
        simulator.run()
    except rospy.ROSInterruptException:
        pass
