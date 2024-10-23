from gen_ref import *
from acados_setting import *
import rospy
from std_msgs.msg import Float64MultiArray
from heron_msgs.msg import Drive
from mpc_msgs.msg import MPCTraj, MPC_State, Obs_State
import math
import time
from DOB import*

# traj_xy = (353126.1653380032, 4026065.6140236068)
# offset = np.array([353126.1653380032, 4026065.6140236068])

traj_xy = (353152, 4026032)
# offset = np.array([353126.1653380032, 4026065.6140236068])
# offset = np.array([353152, 4026032])
offset = np.array([353152, 4026032])
class HeronMPC:
    def __init__(self):
        # ROS settings
        self.rate = rospy.Rate(10)  # 10 Hz for ROS loop rate
        self.ekf_sub = rospy.Subscriber('/ekf/estimated_state', Float64MultiArray, self.ekf_callback, queue_size=100)
        self.weight_sub = rospy.Subscriber('/weight', Float64MultiArray, self.weight_callback, queue_size=100)
        self.thrust_pub = rospy.Publisher('/cmd_drive', Drive)
        self.mpcvis_pub = rospy.Publisher('/mpc_vis', MPCTraj)
        self.traj_pub = rospy.Publisher('/mpc_traj', Float64MultiArray)
        self.L1_pub = rospy.Publisher('/L1_data', Float64MultiArray)  
        
        # Initial states and inputs
        self.x = self.y = self.p = self.u = self.v = self.r = 0.0
        self.n1 = self.n2 = 0.0
        self.states = np.zeros(8)
        
        # MPC parameter settings
        self.Tf = 5 # prediction time 4 sec
        self.N = 50 # prediction horizon
        self.con_dt = 0.1 # control sampling time

        self.head_dist = 1.0
        self.HG = np.zeros(8)
        self.HG[0] = self.head_dist*np.cos(self.p)
        self.HG[1] = self.head_dist*np.sin(self.p)
        self.ocp_solver = setup_trajectory_tracking(self.HG, self.N, self.Tf)

        self.cutoff_frequency = 0.5
        
        # reference trajectory generation
        self.ref_dt = 0.01
        self.ref_iter = 10
        

        self.tfinal=1000.0  # total time duration for the trajectory
        # self.dt=self.ref_dt  # time step for the trajectory
        self.translation=(0, 0)  # translation to apply to the trajectory (tuple of x, y)
        self.theta=0.6*np.pi#np.pi/2  # rotation angle in radians (pi/2 for 90 degrees)
        self.num_s_shapes=10.0  # number of S-shapes (sine wave cycles)
        self.start_point=(30, 15)#(-5, -0)  # starting point of the trajectory
        self.amplitude=20.0  # amplitude of the sine wave
        self.wavelength=10.0  # wavelength of the sine wave
        self.velocity = 0.7

        # # Generate snake-like S-shape trajectory
        # self.reference = generate_snake_s_shape_trajectory(
        #     self.tfinal,  # total time duration for the trajectory
        #     self.ref_dt,  # time step for the trajectory
        #     self.translation,  # translation to apply to the trajectory (tuple of x, y)
        #     self.theta,  # rotation angle in radians (pi/2 for 90 degrees)
        #     self.num_s_shapes,  # number of S-shapes (sine wave cycles)
        #     self.start_point,  # starting point of the trajectory
        #     self.amplitude,  # amplitude of the sine wave
        #     self.wavelength,  # wavelength of the sine wave
        #     self.velocity
        # )
        

        self.reference = generate_figure_eight_trajectory(
            self.tfinal,  # total time duration for the trajectory
            self.ref_dt,  # time step for the trajectory
            self.translation,  # translation to apply to the trajectory (tuple of x, y)
            self.theta,  # rotation angle in radians (pi/2 for 90 degrees)
            self.num_s_shapes,  # number of S-shapes (sine wave cycles)
            self.start_point,  # starting point of the trajectory
            self.amplitude,  # amplitude of the sine wave
            self.wavelength,  # wavelength of the sine wave
            self.velocity
        )
        
        # Downsample the reference trajectory by the iteration factor
        self.reference = self.reference[::self.ref_iter, :]


        self.Q_weight = [2, 2, 0, 1, 1, 0, 1e-4, 1e-4]
        self.R_weight = [1e-3, 1e-3]
         
        # L1 vairable
        self.param_estim = np.array([0.0, 0.0, 0.0])
        self.param_filtered = np.array([0.0, 0.0, 0.0])
        self.state_estim = np.array([0.0, 0.0, 0.0])

        self.disturbnacex = 0.0
        self.disturbnacey = 0.0

        # do some initial iterations to start with a good initial guess (******** todo ********)
        # num_iter_initial = 5
        # for _ in range(num_iter_initial):
        #     ocp_solver.solve_for_x0(x0_bar = x0)
        
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
        
        # self.disturbnacex = msg.data[6]
        # self.disturbnacey = msg.data[7]
        # print(self.states)

    def weight_callback(self, msg):# - frequency = gps callback freq. 
        """Callback to update states from EKF estimated state."""
        self.Q_weight = msg.data[:8]
        self.R_weight = msg.data[8:10]
        self.cutoff_frequency = msg.data[10]
        # print(self.Q_weight)
        # print(self.R_weight)
        # print(self.cutoff_frequency)

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

 
    def run(self):
        k = 0 # -> 현재 시간을 index로 표시 -> 그래야 ref trajectory설정가능(******** todo ********)

        while not rospy.is_shutdown():
            loop_start_time = time.time()
            t = time.time()
            ##### Reference States ######

            self.states = np.array([self.x-offset[0], self.y-offset[1], self.p, self.u, self.v, self.r, self.n1, self.n2])
            self.HG = np.array([self.x + self.head_dist*np.cos(self.p)- offset[0],
                        self.y + self.head_dist*np.sin(self.p)- offset[1],
                        self.p,
                        self.u,
                        self.v,
                        self.r,
                        self.n1,
                        self.n2
                        ])

            for j in range(self.N+1):
                refs = self.reference[k+j,:]
                # refs[2] = self.yaw_discontinuity(refs[2])
                yref = np.hstack((refs[0],refs[1],0,0.0,0, 0,0,0,0,0))
                # print(refs)
                if j == self.N:
                    yref = np.hstack((refs[0],refs[1],0,0.0,0,0,0,0))
                self.ocp_solver.cost_set(j, "yref", yref)


            ##### Obstacle Position ######
            # Unfiltered
            # obs_pos = np.array([self.x+100-offset[0], self.y+100-offset[1], 1,  # Obstacle-1: x, y, radius
            #                     self.x+100-offset[0], self.y+100-offset[1], 1, self.param_estim[0], self.param_estim[1], self.param_estim[2]]) # Obstacle-2: x, y, radius

            du = self.param_filtered[0]*np.cos(self.p) + self.param_filtered[1]*np.sin(self.p)
            dv = -self.param_filtered[0]*np.sin(self.p) + self.param_filtered[1]*np.cos(self.p)
            dr = self.param_filtered[2]

            # du = 0.0
            # dv = 0.0
            # dr = 0.0

            # Filtered
            obs_pos = np.array([self.x+100-offset[0], self.y+100-offset[1], 1,  # Obstacle-1: x, y, radius
                                self.x+100-offset[0], self.y+100-offset[1], 1, du, dv, dr])



            for j in range(self.N+1):
                self.ocp_solver.set(j, "p", obs_pos)
        
            Q_mat = 2*np.diag(self.Q_weight)
            R_mat = 2*np.diag(self.R_weight)

            for j in range(self.N):
                self.ocp_solver.cost_set(j, "W", scipy.linalg.block_diag(Q_mat, R_mat))
            self.ocp_solver.cost_set(self.N, "W", Q_mat)

            # do stuff
            elapsed = time.time() - t
            print(elapsed)
            
            # preparation phase
            self.ocp_solver.options_set('rti_phase', 1)
            status = self.ocp_solver.solve()
            t_preparation = self.ocp_solver.get_stats('time_tot')

            # set initial state
           
            self.ocp_solver.set(0, "lbx", self.HG)
            self.ocp_solver.set(0, "ubx", self.HG)

            # feedback phase
            self.ocp_solver.options_set('rti_phase', 2)
            status = self.ocp_solver.solve()
            t_feedback = self.ocp_solver.get_stats('time_tot')

            st, pf = DOB(self.states, self.state_estim, self.param_filtered, self.con_dt, self.cutoff_frequency)
            self.state_estim, self.param_filtered= st,  pf

            # obtain mpc input
            del_n1_n2 = self.ocp_solver.get(0, "u")
            self.n1 += del_n1_n2[0]*self.con_dt
            self.n2 += del_n1_n2[1]*self.con_dt

            rospy.loginfo("MPC Computation time: %f [sec]", t_preparation + t_feedback)


            drive_msg = Drive()
  
            drive_msg.left = self.force_to_heron_command(self.n1,1)
            drive_msg.right = self.force_to_heron_command(self.n2,2)
  
            self.thrust_pub.publish(drive_msg)                                    
            
            
            # Publish predicted states and reference states
            mpc_data_stack = MPCTraj()
            mpc_data_stack.header.stamp = rospy.Time.now()
            mpc_data_stack.pred_num = self.N
            mpc_data_stack.sampling_time = self.con_dt
            mpc_data_stack.cpu_time = t_preparation + t_feedback	
            
            for j in range(self.N+1):
                mpc_pred = MPC_State()
                mpc_ref = MPC_State()
                mpc_pred.x = self.ocp_solver.get(j, "x")[0] + offset[0]
                mpc_pred.y = self.ocp_solver.get(j, "x")[1] + offset[1]
                # mpc_pred.Vx = 0.0
                # mpc_pred.Vy = 0.0
                # mpc_pred.nx = self.force_to_heron_command(self.ocp_solver.get(j, "x")[6],1)
                # mpc_pred.ny = self.force_to_heron_command(self.ocp_solver.get(j, "x")[7],2)
                mpc_data_stack.state.append(mpc_pred)            
                # print(mpc_pred.u)
                mpc_ref.x = self.reference[k+j,0] + offset[0]
                mpc_ref.y = self.reference[k+j,1] + offset[1]
                # mpc_ref.Vx = self.reference[k+j,2]
                # mpc_ref.Vy = self.reference[k+j,3]
                # mpc_ref.nx = 0.0
                # mpc_ref.ny = 0.0
                mpc_data_stack.ref.append(mpc_ref)            
 
 
            obs_state = Obs_State()
            obs_state.x   = obs_pos[0]+offset[0]
            obs_state.y   = obs_pos[1]+offset[1]
            obs_state.rad = obs_pos[2]
            mpc_data_stack.obs.append(obs_state)
            obs_state = Obs_State()
            obs_state.x   = obs_pos[3]+offset[0]
            obs_state.y   = obs_pos[4]+offset[1]
            obs_state.rad = obs_pos[5]
            mpc_data_stack.obs.append(obs_state)        

            self.mpcvis_pub.publish(mpc_data_stack)
            
            traj_data = [self.tfinal, self.ref_dt, self.theta, self.num_s_shapes, self.start_point[0], self.start_point[1], self.amplitude, self.wavelength, self.velocity, offset[0], offset[1]]
            traj_msg = Float64MultiArray()
            traj_msg.data = traj_data
            self.traj_pub.publish(traj_msg)

            M = 38.0
            # L1_data = [self.param_filtered[0], self.param_filtered[1], self.disturbnacex, self.disturbnacey, self.states[7], self.states[6], thruster[1]+self.n2, thruster[0]+self.n1]
            L1_data = [self.param_filtered[0]/M, self.param_filtered[1]/M, -self.param_filtered[0]/M, -self.param_filtered[1]/M, self.states[7], self.states[6], self.n2, self.n1]
            # print(L1_data)
            L1_msg = Float64MultiArray()
            L1_msg.data = L1_data
            self.L1_pub.publish(L1_msg)


            # Increment the index for the reference trajectory
            k += 1
            if k + self.N >= len(self.reference):
                k = 0  # Reset the index if it goes beyond the reference length
            
            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0, 0.1 - elapsed_time)
            time.sleep(sleep_time)            
            # self.rate.sleep()


if __name__ == '__main__':
    try:
        rospy.init_node('heron_mpc', anonymous=True)
        mpc = HeronMPC()
        rospy.loginfo("Starting MPC control loop.")
        mpc.run()
    except rospy.ROSInterruptException:
        pass