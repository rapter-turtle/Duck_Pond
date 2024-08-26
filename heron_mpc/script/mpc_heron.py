from gen_ref import *
from acados_setting import *
import rospy
from std_msgs.msg import Float64MultiArray
from heron_msgs.msg import Drive
from mpc_msgs.msg import MPCTraj, MPC_State, Obs_State
import math
import time

traj_xy = (353126.1653380032, 4026065.6140236068)
offset = np.array([353126.1653380032, 4026065.6140236068])
class HeronMPC:
    def __init__(self):
        # ROS settings
        self.rate = rospy.Rate(10)  # 10 Hz for ROS loop rate
        self.ekf_sub = rospy.Subscriber('/ekf/estimated_state', Float64MultiArray, self.ekf_callback)
        self.weight_sub = rospy.Subscriber('/weight', Float64MultiArray, self.weight_callback)
        self.thrust_pub = rospy.Publisher('/cmd_drive', Drive)
        self.mpcvis_pub = rospy.Publisher('/mpc_vis', MPCTraj)  
        
        # Initial states and inputs
        self.x = self.y = self.p = self.u = self.v = self.r = 0.0
        self.n1 = self.n2 = 0.0
        self.states = np.zeros(8)
        
        # MPC parameter settings
        self.Tf = 4 # prediction time 4 sec
        self.N = 40 # prediction horizon
        self.con_dt = 0.1 # control sampling time
        self.ocp_solver = setup_trajectory_tracking(self.states, self.N, self.Tf)

        
        # reference trajectory generation
        self.ref_dt = 0.01
        self.ref_iter = int(self.con_dt/self.ref_dt)
        self.reference = generate_figure_eight_trajectory(1000, self.ref_dt, traj_xy, np.pi/2) # reference dt = 0.01 sec, 1000 sec trajectory generation
        # self.reference = generate_figure_eight_trajectory_con(1000, self.ref_dt) # reference dt = 0.01 sec, 1000 sec trajectory generation
        self.reference = self.reference[::self.ref_iter,:]
        
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
        self.states = np.array([self.x-offset[0], self.y-offset[1], self.p, self.u, self.v, self.r, self.n1, self.n2])


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
            t = time.time()
            ##### Reference States ######
            for j in range(self.N+1):
                refs = self.reference[k+j,:]
                refs[2] = self.yaw_discontinuity(refs[2])
                yref = np.hstack((refs[0]-offset[0],refs[1]-offset[1],refs[2],refs[3],0,refs[4],0,0,0,0))
                if j == self.N:
                    yref = np.hstack((refs[0]-offset[0],refs[1]-offset[1],refs[2],refs[3],0,refs[4],0,0))
                self.ocp_solver.cost_set(j, "yref", yref)


            ##### Obstacle Position ######
            obs_pos = np.array([self.x+10-offset[0], self.y+10-offset[1], 1,  # Obstacle-1: x, y, radius
                                self.x+10-offset[0], self.y+10-offset[1], 1]) # Obstacle-2: x, y, radius
            for j in range(self.N+1):
                self.ocp_solver.set(j, "p", obs_pos)
        
            # do stuff
            elapsed = time.time() - t
            print(elapsed)
            
            
            # preparation phase
            self.ocp_solver.options_set('rti_phase', 1)
            status = self.ocp_solver.solve()
            t_preparation = self.ocp_solver.get_stats('time_tot')

            # set initial state
            self.ocp_solver.set(0, "lbx", self.states)
            self.ocp_solver.set(0, "ubx", self.states)

            # feedback phase
            self.ocp_solver.options_set('rti_phase', 2)
            status = self.ocp_solver.solve()
            t_feedback = self.ocp_solver.get_stats('time_tot')

            # obtain mpc input
            del_n1_n2 = self.ocp_solver.get(0, "u")
            self.n1 += del_n1_n2[0]*self.con_dt
            self.n2 += del_n1_n2[1]*self.con_dt
            rospy.loginfo("MPC Computation time: %f [sec]", t_preparation + t_feedback)

            # print(self.states[2])
            # print(yref[2])
            # print(self.n1, self.n2)
            # print(self.states)
            
            # Publish the control inputs (e.g., thrust commands)
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
                mpc_pred.x = self.ocp_solver.get(j, "x")[0]+offset[0]
                mpc_pred.y = self.ocp_solver.get(j, "x")[1]+offset[1]
                mpc_pred.p = self.ocp_solver.get(j, "x")[2]
                mpc_pred.u = self.ocp_solver.get(j, "x")[3]
                mpc_pred.v = self.ocp_solver.get(j, "x")[4]
                mpc_pred.r = self.ocp_solver.get(j, "x")[5]
                mpc_pred.tl = self.force_to_heron_command(self.ocp_solver.get(j, "x")[6],1)
                mpc_pred.tr = self.force_to_heron_command(self.ocp_solver.get(j, "x")[7],2)
                mpc_data_stack.state.append(mpc_pred)            
                # print(mpc_pred.u)
                mpc_ref.x = self.reference[k+j,0]
                mpc_ref.y = self.reference[k+j,1]
                mpc_ref.p = self.reference[k+j,2]
                mpc_ref.u = self.reference[k+j,3]
                mpc_ref.v = 0.0
                mpc_ref.r = self.reference[k+j,4]
                mpc_ref.tl = 0.0
                mpc_ref.tr = 0.0
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
            
            # Increment the index for the reference trajectory
            k += 1
            if k + self.N >= len(self.reference):
                k = 0  # Reset the index if it goes beyond the reference length
            self.rate.sleep()


if __name__ == '__main__':
    try:
        rospy.init_node('heron_mpc', anonymous=True)
        mpc = HeronMPC()
        rospy.loginfo("Starting MPC control loop.")
        mpc.run()
    except rospy.ROSInterruptException:
        pass