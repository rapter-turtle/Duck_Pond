from plot_vehicle import *
from acados_setting import *
from load_param import *
from kinematic_integrator import *
import pandas as pd
import matplotlib.pyplot as plt

def main(cbf_num,mode,prediction_horizon,gamma2):
    vehicle_p = load_kinematic_param
    vehicle_p.N = prediction_horizon

    N_horizon = vehicle_p.N
    dt = vehicle_p.dt    
    con_dt = dt
    vehicle_p.CBF = cbf_num
    vehicle_p.gamma2 = gamma2
    vehicle_p.gamma_TC2 = gamma2
    
    # vehicle_p.TCCBF = tccbf_type
    # vehicle_p.gamma1 = gamma1
    if mode == 'static_straight':
        Tf = 40
        target_x = 40
    if mode == 'static_narrow':   
        Tf = 12
        target_x = 20
    if mode == 'avoid':   
        Tf = 20*1
        target_x = 35
        # vehicle_p.gamma2 = 0.03
        # vehicle_p.gamma_TC2 = 0.04
    if mode == 'overtaking':   
        Tf = 20
        target_x = 35
        # vehicle_p.gamma2 = 0.25
        # vehicle_p.gamma_TC2 = 0.25
    if mode == 'single_static_straight':
        Tf = 25
        target_x = 35

    Nsim = int(Tf/dt)
    # Initial state
    target_speed = 2.0
    
    obsrad_param = 1/5

    x0 = np.array([0.0 , # x
                   0.0, # y
                   0.0, # psi
                   target_speed, # surge
                   0.0, # rot
                   0.0])  # dacc    

    ocp_solver, integrator = setup_wpt_tracking(x0,mode)

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    simX = np.zeros((Nsim+1, nx))
    simU = np.zeros((Nsim+1, nu))
    simX[0,:] = x0
    yref_list = []
    obs_list = []
    mpc_pred_list = []
    cbf_and_dist = np.zeros((Nsim+1, 10)) # cbf 1~5, dist 1~5

    t_preparation = np.zeros((Nsim+1))
    t_feedback = np.zeros((Nsim+1))

    # do some initial iterations to start with a good initial guess
    # num_iter_initial = 25
    # for _ in range(num_iter_initial):
    #     ocp_solver.solve_for_x0(x0_bar = x0)

    # closed loop
    
    for i in range(Nsim):
        # print(i)
        print('CBF Num:' + str(cbf_num) + ', N:' + str(prediction_horizon) + ', Mode:' + str(mode) + ', Iter:' + str(i+1) + '/'+ str(Nsim))
        
        for j in range(N_horizon):
            yref = np.hstack((0,0,0,target_speed,0,0,0,0))
            ocp_solver.cost_set(j, "yref", yref)
        yref_N = np.hstack((0,0,0,target_speed,0,0))
        ocp_solver.cost_set(N_horizon, "yref", yref_N)

        

        ## static obstacles - Straight
        if mode == 'static_straight' or mode == 'single_static_straight':
            ox1 = 15; oy1 = +0.05; or1 = 10.0*obsrad_param
            ox2 = 37; oy2 = +0.05; or2 = 15.0*obsrad_param
            ox3 = 50; oy3 = +20.05; or3 = 1.0*obsrad_param
            ox4 = 50; oy4 = +20.05; or4 = 1.0*obsrad_param
            ox5 = 50; oy5 = +20.05; or5 = 1.0*obsrad_param
            obs_index = 2
            if mode == 'single_static_straight':
                ox1 = 15; oy1 = +0.01; or1 = 10.0*obsrad_param
                ox2 = 15; oy2 = +0.01; or2 = 10.0*obsrad_param
                ox3 = 15; oy3 = +0.01; or3 = 10.0*obsrad_param
                ox4 = 15; oy4 = +0.01; or4 = 10.0*obsrad_param
                ox5 = 15; oy5 = +0.01; or5 = 10.0*obsrad_param
                obs_index = 1

            obs_pos = np.array([ox1, oy1, or1 + vehicle_p.radius, 0, 0,  
                                ox2, oy2, or2 + vehicle_p.radius, 0, 0,  
                                ox3, oy3, or3 + vehicle_p.radius, 0, 0,  
                                ox4, oy4, or4 + vehicle_p.radius, 0, 0,  
                                ox5, oy5, or5 + vehicle_p.radius, 0, 0])    
        
            for jj in range(5):
                cbf_and_dist[i,jj] = calc_cbf(simX[i,:],obs_pos[5*jj+0:5*jj+5],cbf_num)
                cbf_and_dist[i,5+jj] = calc_closest_distance(simX[i,:],obs_pos[5*jj+0:5*jj+5])
        
            obs_list.append(obs_pos)

 
                
            for j in range(N_horizon):
                ocp_solver.set(j, "p", obs_pos)
            ocp_solver.set(N_horizon, "p", obs_pos)

            
        ## Static obstacles - Narrow
        if mode == 'static_narrow':
            ox1 = 12; oy1 = +3; or1 = 10.0*obsrad_param
            ox2 = 12; oy2 = -3; or2 = 10.0*obsrad_param
            ox3 = 12; oy3 = +3; or3 = 10.0*obsrad_param
            ox4 = 12; oy4 = +3; or4 = 10.0*obsrad_param
            ox5 = 12; oy5 = -3; or5 = 10.0*obsrad_param
            obs_index = 2
            obs_pos = np.array([ox1, oy1, or1 + vehicle_p.radius, 0, 0,  
                                ox2, oy2, or2 + vehicle_p.radius, 0, 0,  
                                ox3, oy3, or3 + vehicle_p.radius, 0, 0,  
                                ox4, oy4, or4 + vehicle_p.radius, 0, 0,  
                                ox5, oy5, or5 + vehicle_p.radius, 0, 0])    
        
            obs_list.append(obs_pos)
            
            for jj in range(5):
                cbf_and_dist[i,jj] = calc_cbf(simX[i,:],obs_pos[5*jj+0:5*jj+5],cbf_num)
                cbf_and_dist[i,5+jj] = calc_closest_distance(simX[i,:],obs_pos[5*jj+0:5*jj+5])    
                    
            for j in range(N_horizon):
                ocp_solver.set(j, "p", obs_pos)
            ocp_solver.set(N_horizon, "p", obs_pos)

        ## Dynamic obstacles - Avoid              
        if mode == 'avoid':
            for j in range(N_horizon+1):
                obs_speed = -0.5
                xinit = 20
                ox1 = xinit + (i+j)*dt*obs_speed;  
                oy1 = 0.001; 
                or1 = 10.0*obsrad_param
                ox2 = xinit + (i+j)*dt*obs_speed;  
                oy2 = 0.001; 
                or2 = 10.0*obsrad_param
                ox3 = xinit + (i+j)*dt*obs_speed;  
                oy3 = 0.001; 
                or3 = 10.0*obsrad_param
                ox4 = xinit + (i+j)*dt*obs_speed;  
                oy4 = 0.001; 
                or4 = 10.0*obsrad_param
                ox5 = xinit + (i+j)*dt*obs_speed;  
                oy5 = 0.001; 
                or5 = 10.0*obsrad_param
                obs_index = 1

                obs_pos = np.array([ox1, oy1, or1 + vehicle_p.radius, obs_speed, 0,  
                                    ox2, oy2, or2 + vehicle_p.radius, obs_speed, 0,  
                                    ox3, oy3, or3 + vehicle_p.radius, obs_speed, 0,  
                                    ox4, oy4, or4 + vehicle_p.radius, obs_speed, 0,  
                                    ox5, oy5, or5 + vehicle_p.radius, obs_speed, 0])    
                obs_pos_ignore = np.array([ox1, oy1, 0.00001, 0, 0,  
                                            ox2, oy2, 0.00001, 0, 0,  
                                            ox3, oy3, 0.00001, 0, 0,  
                                            ox4, oy4, 0.00001, 0, 0,  
                                            ox5, oy5, 0.00001, 0, 0])  
                if j == 0:
                    for jj in range(5):
                        cbf_and_dist[i,jj] = calc_cbf(simX[i,:],obs_pos[5*jj+0:5*jj+5],cbf_num)
                        cbf_and_dist[i,5+jj] = calc_closest_distance(simX[i,:],obs_pos[5*jj+0:5*jj+5])                

                if j == 0:
                    obs_list.append(obs_pos)
 

                # if simX[i, 0] > ox5:
                #     obs_pos = np.array([ox1, oy1, 0.00001, 0, 0,  
                #                         ox2, oy2, 0.00001, 0, 0,  
                #                         ox3, oy3, 0.00001, 0, 0,  
                #                         ox4, oy4, 0.00001, 0, 0,  
                #                         ox5, oy5, 0.00001, 0, 0])  
                if j == N_horizon:
                    if np.abs(np.arctan2((oy1-simX[i, 1]),(ox1-simX[i, 0]))-simX[i, 2])>np.pi/2 or simX[i, 0] > ox5:
                        ocp_solver.set(N_horizon, "p", obs_pos_ignore)
                    else:
                        ocp_solver.set(N_horizon, "p", obs_pos)
                else:
                    if np.abs(np.arctan2((oy1-simX[i, 1]),(ox1-simX[i, 0]))-simX[i, 2])>np.pi/2 or simX[i, 0] > ox5:
                        ocp_solver.set(j, "p", obs_pos_ignore)         
                    else:
                        ocp_solver.set(j, "p", obs_pos)         


        ## Dynamic obstacles - Overtaking              
        if mode == 'overtaking':
            for j in range(N_horizon+1):
                obs_speed = 0.5
                obs_init = 10
                ox1 = obs_init + (i+j)*dt*obs_speed; 
                oy1 = 0.001; 
                or1 = 10.0*obsrad_param
                ox2 = obs_init + (i+j)*dt*obs_speed; 
                oy2 = 0.001; 
                or2 = 10.0*obsrad_param
                ox3 = obs_init + (i+j)*dt*obs_speed; 
                oy3 = 0.001; 
                or3 = 10.0*obsrad_param
                ox4 = obs_init + (i+j)*dt*obs_speed; 
                oy4 = 0.001; 
                or4 = 10.0*obsrad_param
                ox5 = obs_init + (i+j)*dt*obs_speed; 
                oy5 = 0.001; 
                or5 = 10.0*obsrad_param
                obs_index = 1

                obs_pos = np.array([ox1, oy1, or1 + vehicle_p.radius, obs_speed, 0,  
                                    ox2, oy2, or2 + vehicle_p.radius, obs_speed, 0,  
                                    ox3, oy3, or3 + vehicle_p.radius, obs_speed, 0,  
                                    ox4, oy4, or4 + vehicle_p.radius, obs_speed, 0,  
                                    ox5, oy5, or5 + vehicle_p.radius, obs_speed, 0])    
                if j == 0:
                    for jj in range(5):
                        cbf_and_dist[i,jj] = calc_cbf(simX[i,:],obs_pos[5*jj+0:5*jj+5],cbf_num)
                        cbf_and_dist[i,5+jj] = calc_closest_distance(simX[i,:],obs_pos[5*jj+0:5*jj+5])                
                
                if j == 0:
                    obs_list.append(obs_pos)
                    

                # if simX[i, 0] > (i)*dt*obs_speed:
                #     obs_pos = np.array([ox1, oy1, 0.00001, obs_speed, 0,  
                #                         ox2, oy2, 0.00001, obs_speed, 0,  
                #                         ox3, oy3, 0.00001, obs_speed, 0,  
                #                         ox4, oy4, 0.00001, obs_speed, 0,  
                #                         ox5, oy5, 0.00001, obs_speed, 0])                       
                ocp_solver.set(j, "p", obs_pos)         



        # preparation phase
        ocp_solver.options_set('rti_phase', 1)
        status = ocp_solver.solve()
        t_preparation[i] = ocp_solver.get_stats('time_tot')

        # set initial state
        ocp_solver.set(0, "lbx", simX[i, :])
        ocp_solver.set(0, "ubx", simX[i, :])

        # feedback phase
        ocp_solver.options_set('rti_phase', 2)
        status = ocp_solver.solve()
        t_feedback[i] = ocp_solver.get_stats('time_tot')

        simU[i, :] = ocp_solver.get(0, "u")
        
        # print((t_preparation[i] + t_feedback[i])*1000)

        mpc_pred = []
        for j in range(N_horizon+1):
            mpc_pred.append(ocp_solver.get(j, "x")[0:2]) 
        mpc_pred_array = np.vstack(mpc_pred)
        mpc_pred_list.append(mpc_pred_array)

        # simulate system
        # simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])
        simX[i+1, :] = kinematic_integrator(simX[i, :], simU[i,:], dt)

        # print(simX[i, 3])
        # print(simU[i, :])
    simU[i+1, :] = simU[i, :]
    cbf_and_dist[i+1, :] = cbf_and_dist[i, :]
    yref_list.append(yref)
    mpc_pred_list.append(mpc_pred_array)
    obs_list.append(obs_pos)

    # evaluate timings
    # scale to milliseconds
    t_preparation *= 1000
    t_feedback *= 1000
    print(f'Computation time in preparation phase in ms: \
            min {np.min(t_preparation):.3f} median {np.median(t_preparation):.3f} max {np.max(t_preparation):.3f}')
    print(f'Computation time in feedback phase in ms:    \
            min {np.min(t_feedback):.3f} median {np.median(t_feedback):.3f} max {np.max(t_feedback):.3f}')
    
    yref_array = np.array(yref_list)
    obs_array = np.array(obs_list)

    ocp_solver = None
    plot_iter = len(simX)-1
    # plot_iter = 10
    animateASV(simX, simU, target_speed, mpc_pred_list, obs_array, cbf_and_dist, plot_iter, t_preparation+t_feedback, mode, obs_index,target_x)

    # animateCBF(plot_iter, cbf_and_dist, mode)

    # t = np.arange(0, con_dt*Nsim, con_dt)
    # plot_inputs(t, simX, simU, target_speed, mode)
    

    ## calculate cost
    closest_dist = np.min((cbf_and_dist[:,5],cbf_and_dist[:,6],cbf_and_dist[:,7],cbf_and_dist[:,8],cbf_and_dist[:,9]))
    t_ap = 0
    for i in range(Nsim):
        if simX[i,0] < target_x:
            t_ap += dt
    arrival_time = t_ap
    speed_error = simX[:,3] - target_speed
    speed_var = np.var(speed_error)
    rot_usage = np.sum(simX[:,4]**2)
    acc_usage =  np.sum(simX[:,5]**2)
    drot_usage = np.sum(simU[:,0]**2)
    dacc_usage = np.sum(simU[:,1]**2)

    return closest_dist, arrival_time , speed_var ,rot_usage ,acc_usage ,drot_usage ,dacc_usage 



def calc_closest_distance(state,obs):
    x = state[0]; y = state[1]
    ox = obs[0]; oy = obs[1]; orad = obs[2]
    dist = np.sqrt( (x-ox)**2 + (y - oy)**2) - orad
    return dist

def calc_cbf(state,obs,type):
    x = state[0]; y = state[1]
    psi = state[2]
    u = state[3]
    ox = obs[0]; oy = obs[1]; orad = obs[2]
    vehicle_p = load_kinematic_param

    if type == 0:    
        cbf = np.sqrt( (x-ox)**2 + (y - oy)**2) - orad

    if type == 1:    
        B = np.sqrt( (x-ox)**2 + (y - oy)**2) - orad
        Bdot = ((x-ox)*(u*cos(psi)) + (y-oy)*(u*sin(psi)))/np.sqrt((x-ox)**2 + (y - oy)**2)
        cbf = Bdot + vehicle_p.gamma1*B

    if type == 2:
        R = u/vehicle_p.rmax*vehicle_p.gamma_TC1
        # R = 10
        B1 = np.sqrt( (ox-x-R*cos(psi-np.pi/2))**2 + (oy-y-R*sin(psi-np.pi/2))**2) - (orad+R)
        B2 = np.sqrt( (ox-x-R*cos(psi+np.pi/2))**2 + (oy-y-R*sin(psi+np.pi/2))**2) - (orad+R)
        if vehicle_p.TCCBF == 3:
            cbf = np.log((np.exp(B1)+np.exp(B2)-1))
        if vehicle_p.TCCBF == 2:
            cbf = B2
        if vehicle_p.TCCBF == 1:
            cbf = B1

    return cbf

if __name__ == '__main__':



    g2 = 0.1
    g2_tc = 0.1
    cd1, at1 ,sv1 ,ru1 ,au1 ,dru1 ,dau1  = main(1,'avoid', 10, 0.03)
    cd2, at2 ,sv2 ,ru2 ,au2 ,dru2 ,dau2  = main(2,'avoid', 10, 0.03)
    # cd3, at3 ,sv3 ,ru3 ,au3 ,dru3 ,dau3  = main(1,'overtaking',10, g2)
    # cd4, at4 ,sv4 ,ru4 ,au4 ,dru4 ,dau4  = main(2,'overtaking',10, g2_tc)
    # cd5, at5 ,sv5 ,ru5 ,au5 ,dru5 ,dau5  = main(1,'static_narrow',10, g2)
    # cd6, at6 ,sv6 ,ru6 ,au6 ,dru6 ,dau6  = main(2,'static_narrow',10, g2_tc)
    # cd7, at7 ,sv7 ,ru7 ,au7 ,dru7 ,dau7  = main(1,'single_static_straight',10, g2)
    # cd8, at8 ,sv8 ,ru8 ,au8 ,dru8 ,dau8  = main(2,'single_static_straight',10, g2_tc)

    # plt.pause(0.01)
    # plt.close('all')
    
    # # Sample data based on the provided outputs for each case
    # data = {
    #     "Case": [
    #         "1 - avoid", "2 - avoid",
    #         "1 - overtaking", "2 - overtaking",
    #         "1 - static_narrow", "2 - static_narrow",
    #         "1 - single_static_straight", "2 - single_static_straight"
    #     ],
    #     "closest_dist": [cd1,cd2,cd3,cd4,cd5,cd6,cd7,cd8],
    #     "arrival_time":[at1, at2, at3, at4, at5, at6, at7, at8],
    #     "speed_var":  [sv1, sv2, sv3, sv4, sv5, sv6, sv7, sv8],
    #     "rot_usage":  [ru1, ru2, ru3, ru4, ru5, ru6, ru7, ru8],
    #     "acc_usage":  [au1, au2, au3, au4, au5, au6, au7, au8],
    #     "drot_usage": [dru1, dru2, dru3, dru4, dru5, dru6, dru7, dru8],
    #     "dacc_usage": [dau1, dau2, dau3, dau4, dau5, dau6, dau7, dau8],
    # }

    # # Create DataFrame
    # df = pd.DataFrame(data)
    # df = df.round(3)

    # # Create a figure and axis
    # fig, ax = plt.subplots(figsize=(12, 8))

    # # Hide the axes
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    # ax.set_frame_on(False)
    # # Create the table
    # table = pd.plotting.table(ax, df, loc='center', cellLoc='center', colWidths=[0.2]*len(df.columns))
    # # Style the table
    # table.auto_set_font_size(False)
    # table.set_fontsize(13)
    # table.auto_set_column_width(col=list(range(len(df.columns))))

    # # Adjust layout
    # plt.subplots_adjust(left=0.3, right=0.7, top=0.9, bottom=0.1)

    # # Show the table
    # plt.show()