from plot_ship import *
from acados_setting import *
from load_param import *
from ship_integrator import *

import pandas as pd
import matplotlib.pyplot as plt

def main(cbf_num,mode,prediction_horizon,gamma2):
    ship_p = load_ship_param
    ship_p.N = prediction_horizon

    N_horizon = ship_p.N
    dt = ship_p.dt    
    con_dt = dt
    ship_p.CBF = cbf_num
    ship_p.gamma2 = gamma2
    ship_p.gamma_TC2 = gamma2
    
    # ship_p.TCCBF = tccbf_type
    # ship_p.gamma1 = gamma1
    if mode == 'static_straight':
        Tf = 60        
    if mode == 'static_narrow':   
        Tf = 25
        target_x = 20
    if mode == 'avoid':   
        Tf = 70
        target_x = 40
        # ship_p.gamma2 = 0.03
        # ship_p.gamma_TC2 = 0.04
    if mode == 'overtaking':   
        Tf = 65
        target_x = 40
        # ship_p.gamma2 = 0.25
        # ship_p.gamma_TC2 = 0.25
    if mode == 'single_static_straight':
        Tf = 50
        target_x = 35
    
    if mode == 'param_tuning':
        Tf = 100
        target_x = 50

    Nsim = int(Tf/dt)
    # Initial state
    target_speed = 1.0
    
    obsrad_param = 1/2

    Fx_init = ship_p.Xu*target_speed + ship_p.Xuu*target_speed**2
    x0 = np.array([0.0 , # x
                   0.0, # y
                   0.0, # psi
                   target_speed, # surge
                   0.0, # sway
                   0.0, # rot-vel
                   Fx_init/2,  # Fx
                   Fx_init/2])  # Fn    
    
    if mode == 'param_tuning':
        x0 = np.array([0.0 , # x
                    15.0, # y
                    0.0, # psi
                    target_speed, # surge
                    0.0, # sway
                    0.0, # rot-vel
                    Fx_init/2,  # Fx
                    Fx_init/2])  # Fn    
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
            yref = np.hstack((0,0,0,target_speed,target_speed,0,Fx_init/2,Fx_init/2,0,0))
            ocp_solver.cost_set(j, "yref", yref)
        yref_N = np.hstack((0,0,0,target_speed,target_speed,0,Fx_init/2,Fx_init/2))
        ocp_solver.cost_set(N_horizon, "yref", yref_N)

        if mode == 'param_tuning':
            ox1 = 30; oy1 = 25; or1 = 0.0001
            ox2 = 30; oy2 = 25; or2 = 0.0001
            ox3 = 30; oy3 = 25; or3 = 0.0001
            ox4 = 30; oy4 = 25; or4 = 0.0001
            ox5 = 30; oy5 = 25; or5 = 0.0001
            obs_index = 1

            obs_pos = np.array([ox1, oy1, or1 + ship_p.radius, 0, 0,  
                                ox2, oy2, or2 + ship_p.radius, 0, 0,  
                                ox3, oy3, or3 + ship_p.radius, 0, 0,  
                                ox4, oy4, or4 + ship_p.radius, 0, 0,  
                                ox5, oy5, or5 + ship_p.radius, 0, 0])    
        
            for jj in range(5):
                cbf_and_dist[i,jj] = calc_cbf(simX[i,:],obs_pos[5*jj+0:5*jj+5],cbf_num)
                cbf_and_dist[i,5+jj] = calc_closest_distance(simX[i,:],obs_pos[5*jj+0:5*jj+5])
            obs_list.append(obs_pos)
            for j in range(N_horizon):
                ocp_solver.set(j, "p", obs_pos)
            ocp_solver.set(N_horizon, "p", obs_pos)



        ## static obstacles - Straight
        if mode == 'static_straight' or mode == 'single_static_straight':
            ox1 = 15; oy1 = +0.05; or1 = 10.0*obsrad_param
            ox2 = 37; oy2 = +0.05; or2 = 15.0*obsrad_param
            ox3 = 50; oy3 = +20.05; or3 = 1.0*obsrad_param
            ox4 = 50; oy4 = +20.05; or4 = 1.0*obsrad_param
            ox5 = 50; oy5 = +20.05; or5 = 1.0*obsrad_param
            obs_index = 2
            if mode == 'single_static_straight':
                ox1 = 1*25; oy1 = +0.01; or1 = 10.0*obsrad_param
                ox2 = 1*25; oy2 = +0.01; or2 = 10.0*obsrad_param
                ox3 = 1*25; oy3 = +0.01; or3 = 10.0*obsrad_param
                ox4 = 1*25; oy4 = +0.01; or4 = 10.0*obsrad_param
                ox5 = 1*25; oy5 = +0.01; or5 = 10.0*obsrad_param
                obs_index = 1

            obs_pos = np.array([ox1, oy1, or1 + ship_p.radius, 0, 0,  
                                ox2, oy2, or2 + ship_p.radius, 0, 0,  
                                ox3, oy3, or3 + ship_p.radius, 0, 0,  
                                ox4, oy4, or4 + ship_p.radius, 0, 0,  
                                ox5, oy5, or5 + ship_p.radius, 0, 0])    
        
            for jj in range(5):
                cbf_and_dist[i,jj] = calc_cbf(simX[i,:],obs_pos[5*jj+0:5*jj+5],cbf_num)
                cbf_and_dist[i,5+jj] = calc_closest_distance(simX[i,:],obs_pos[5*jj+0:5*jj+5])
        
            obs_list.append(obs_pos)

 
                
            for j in range(N_horizon):
                ocp_solver.set(j, "p", obs_pos)
            ocp_solver.set(N_horizon, "p", obs_pos)

            
        ## Static obstacles - Narrow
        if mode == 'static_narrow':
            ox1 = 12; oy1 = +10.0*obsrad_param+2; or1 = 10.0*obsrad_param
            ox2 = 12; oy2 = -10.0*obsrad_param-2; or2 = 10.0*obsrad_param
            ox3 = 12; oy3 = +10.0*obsrad_param+2; or3 = 10.0*obsrad_param
            ox4 = 12; oy4 = +10.0*obsrad_param+2; or4 = 10.0*obsrad_param
            ox5 = 12; oy5 = -10.0*obsrad_param-2; or5 = 10.0*obsrad_param
            obs_index = 2
            obs_pos = np.array([ox1, oy1, or1 + ship_p.radius, 0, 0,  
                                ox2, oy2, or2 + ship_p.radius, 0, 0,  
                                ox3, oy3, or3 + ship_p.radius, 0, 0,  
                                ox4, oy4, or4 + ship_p.radius, 0, 0,  
                                ox5, oy5, or5 + ship_p.radius, 0, 0])    
        
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
                obs_speed = -0.3
                xinit = 35
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

                obs_pos = np.array([ox1, oy1, or1 + ship_p.radius, obs_speed, 0,  
                                    ox2, oy2, or2 + ship_p.radius, obs_speed, 0,  
                                    ox3, oy3, or3 + ship_p.radius, obs_speed, 0,  
                                    ox4, oy4, or4 + ship_p.radius, obs_speed, 0,  
                                    ox5, oy5, or5 + ship_p.radius, obs_speed, 0])    
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
                obs_speed = 0.3
                obs_init = 15
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

                obs_pos = np.array([ox1, oy1, or1 + ship_p.radius, obs_speed, 0,  
                                    ox2, oy2, or2 + ship_p.radius, obs_speed, 0,  
                                    ox3, oy3, or3 + ship_p.radius, obs_speed, 0,  
                                    ox4, oy4, or4 + ship_p.radius, obs_speed, 0,  
                                    ox5, oy5, or5 + ship_p.radius, obs_speed, 0])    
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
        simX[i+1, :] = ship_integrator(simX[i, :], simU[i,:], dt)

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
    # print(f'Computation time in preparation phase in ms: \
    #         min {np.min(t_preparation):.3f} median {np.median(t_preparation):.3f} max {np.max(t_preparation):.3f}')
    # print(f'Computation time in feedback phase in ms:    \
    #         min {np.min(t_feedback):.3f} median {np.median(t_feedback):.3f} max {np.max(t_feedback):.3f}')
    yref_array = np.array(yref_list)
    obs_array = np.array(obs_list)

    ocp_solver = None
    plot_iter = len(simX)-1
    # plot_iter = 30

    animateASV(simX, simU, target_speed, mpc_pred_list, obs_array, cbf_and_dist, plot_iter, t_preparation+t_feedback, mode, obs_index)

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
    rot_usage = np.sum(simX[:,5]**2)
    FX_usage =  np.sum((simX[:,6]-Fx_init/2)**2) + np.sum((simX[:,7]-Fx_init/2)**2)
    dFX_usage = np.sum(simU[:,0]**2) + np.sum(simU[:,1]**2)
    
    return closest_dist, arrival_time , speed_var ,rot_usage ,FX_usage , dFX_usage


def calc_closest_distance(state,obs):
    x = state[0]; y = state[1]
    ox = obs[0]; oy = obs[1]; orad = obs[2]
    dist = np.sqrt( (x-ox)**2 + (y - oy)**2) - orad
    return dist

def calc_cbf(state,obs,type):
    x = state[0]; y = state[1]
    psi = state[2]
    u = state[3]
    v = state[4]
    ox = obs[0]; oy = obs[1]; orad = obs[2]
    ship_p = load_ship_param

    if type == 0:    
        cbf = np.sqrt( (x-ox)**2 + (y - oy)**2) - orad

    if type == 1:    
        B = np.sqrt( (x-ox)**2 + (y - oy)**2) - orad
        Bdot = ((x-ox)*(u*cos(psi)+v*sin(psi)) + (y-oy)*(u*sin(psi)-v*cos(psi)))/np.sqrt((x-ox)**2 + (y - oy)**2)
        cbf = Bdot + ship_p.gamma1/orad*B

    if type == 2:
        R = u/ship_p.rmax*ship_p.gamma_TC1
        # R = 10
        B1 = np.sqrt( (ox-x-R*cos(psi-np.pi/2))**2 + (oy-y-R*sin(psi-np.pi/2))**2) - (orad+R)
        B2 = np.sqrt( (ox-x-R*cos(psi+np.pi/2))**2 + (oy-y-R*sin(psi+np.pi/2))**2) - (orad+R)
        if ship_p.TCCBF == 3:
            cbf = np.log((np.exp(B1)+np.exp(B2)-1))
        if ship_p.TCCBF == 2:
            cbf = B2
        if ship_p.TCCBF == 1:
            cbf = B1


    if type == 3:
        R = np.sqrt(u**2+v**2)/ship_p.rmax*ship_p.gamma_TC1
        # R = 10
        B1 = np.sqrt( (ox-x-R*cos(psi-np.arctan(v/u)-np.pi/2))**2 + (oy-y-R*sin(psi-np.arctan(v/u)-np.pi/2))**2) - (orad+R)
        B2 = np.sqrt( (ox-x-R*cos(psi-np.arctan(v/u)+np.pi/2))**2 + (oy-y-R*sin(psi-np.arctan(v/u)+np.pi/2))**2) - (orad+R)
        if ship_p.TCCBF == 3:
            cbf = np.log((np.exp(B1)+np.exp(B2)-1))
        if ship_p.TCCBF == 2:
            cbf = B2
        if ship_p.TCCBF == 1:
            cbf = B1

    return cbf

if __name__ == '__main__':
    # main(1,'param_tuning',10,0.1)
    cd2, at2 ,sv2 ,ru2 ,FXu1 , dFX1  = main(1,'single_static_straight',10, 0.02)
    # cd2, at2 ,sv2 ,ru2 ,FXu1 , dFX1  = main(2,'avoid',10, 0.01)
    # cd2, at2 ,sv2 ,ru2 ,FXu2 , dFX2  = main(3,'avoid',10, 0.01)
    # print(FXu1 , dFX1)
    # print(FXu2 , dFX2)
    plt.show()
    

    go1 = 0
    go2 = 0
    go3 = 0
    
    if go1:
        g2_avoid = 0.008
        cd1, at1 ,sv1 ,ru1 ,FXu1 , dFX1  = main(1,'avoid',10, g2_avoid)

        # g2 = 0.01
        # cd3, at3 ,sv3 ,ru3 ,FXu3 , dFX3  = main(1,'overtaking',10,g2)
        # cd5, at5 ,sv5 ,ru5 ,FXu5 , dFX5  = main(1,'single_static_straight',10,g2)
        
    if go2:
        g2_tc_avoid = 0.015
        cd2, at2 ,sv2 ,ru2 ,FXu2 , dFX2  = main(3,'avoid',10, g2_tc_avoid)

        # g2_tc = 0.02
        # cd4, at4 ,sv4 ,ru4 ,FXu4 , dFX4  = main(3,'overtaking',10,g2_tc)
        # cd6, at6 ,sv6 ,ru6 ,FXu6 , dFX6  = main(3,'single_static_straight',10,g2_tc)
        
    if go3:
        plt.pause(0.01)
        plt.close('all')
        
        # Sample data based on the provided outputs for each case
        data = {
            "Case": [
                "1 - avoid", "2 - avoid",
                "1 - overtaking", "2 - overtaking",
                "1 - single_static_straight", "2 - single_static_straight",
            ],
            "closest_dist": [cd1,cd2,cd3,cd4,cd5,cd6],
            "arrival_time":[at1, at2, at3, at4, at5, at6],
            "speed_var":  [sv1, sv2, sv3, sv4, sv5, sv6],
            "rot_usage":  [ru1, ru2, ru3, ru4, ru5, ru6],
            "FX_usage":  [FXu1, FXu2, FXu3, FXu4, FXu5, FXu6],
            "dFX_usage": [dFX1, dFX2, dFX3, dFX4, dFX5, dFX6],
        }

        # Create DataFrame
        df = pd.DataFrame(data)
        df = df.round(3)

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Hide the axes
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)
        # Create the table
        table = pd.plotting.table(ax, df, loc='center', cellLoc='center', colWidths=[0.2]*len(df.columns))
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.auto_set_column_width(col=list(range(len(df.columns))))

        # Adjust layout
        plt.subplots_adjust(left=0.3, right=0.7, top=0.9, bottom=0.1)

        # Show the table
        plt.show()