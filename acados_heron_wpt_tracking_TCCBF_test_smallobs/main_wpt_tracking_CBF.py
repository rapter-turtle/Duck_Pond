from plot_ship import *
from acados_setting import *
from load_param import *
from ship_integrator import *

def main(cbf_num,mode,prediction_horizon,gamma1=1.5):
    ship_p = load_ship_param
    ship_p.N = prediction_horizon

    N_horizon = ship_p.N
    dt = ship_p.dt    
    con_dt = dt
    ship_p.CBF = cbf_num
    # ship_p.TCCBF = tccbf_type
    ship_p.gamma1 = gamma1
    if mode == 'static_straight':
        Tf = 40
    if mode == 'static_narrow':   
        Tf = 15
    if mode == 'avoid':   
        Tf = 30
    if mode == 'overtaking':   
        Tf = 25
    if mode == 'single_static_straight':
        Tf = 30

    Nsim = int(Tf/dt)
    # Initial state
    target_speed = 1.5
    
    obsrad_param = 1/5

    Fx_init = ship_p.Xu*target_speed + ship_p.Xuu*target_speed**2
    x0 = np.array([0.0 , # x
                   0.0, # y
                   0.0, # psi
                   target_speed, # vel
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
            yref = np.hstack((0,0,0,target_speed,0,Fx_init/2,Fx_init/2,0,0))
            ocp_solver.cost_set(j, "yref", yref)
        yref_N = np.hstack((0,0,0,target_speed,0,Fx_init/2,Fx_init/2))
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
                ox1 = 1*20; oy1 = +0.01; or1 = 15.0*obsrad_param
                ox2 = 1*20; oy2 = +0.01; or2 = 15.0*obsrad_param
                ox3 = 1*20; oy3 = +0.01; or3 = 15.0*obsrad_param
                ox4 = 1*20; oy4 = +0.01; or4 = 15.0*obsrad_param
                ox5 = 1*20; oy5 = +0.01; or5 = 15.0*obsrad_param
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
            ox1 = 12; oy1 = +3; or1 = 10.0*obsrad_param
            ox2 = 12; oy2 = -3; or2 = 10.0*obsrad_param
            ox3 = 12; oy3 = +3; or3 = 10.0*obsrad_param
            ox4 = 12; oy4 = +3; or4 = 10.0*obsrad_param
            ox5 = 12; oy5 = -3; or5 = 10.0*obsrad_param
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
                obs_speed = -0.5
                xinit = 30
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
    # plot_iter = 50
    animateASV(simX, simU, target_speed, mpc_pred_list, obs_array, cbf_and_dist, plot_iter, t_preparation+t_feedback, mode, obs_index)

    # animateCBF(plot_iter, cbf_and_dist, mode)

    # t = np.arange(0, con_dt*Nsim, con_dt)
    # plot_inputs(t, simX, simU, target_speed, mode)

def calc_closest_distance(state,obs):
    x = state[0]; y = state[1]
    ox = obs[0]; oy = obs[1]; orad = obs[2]
    dist = np.sqrt( (x-ox)**2 + (y - oy)**2) - orad
    return dist

def calc_cbf(state,obs,type):
    x = state[0]; y = state[1]
    psi = state[2]; v = state[3]
    ox = obs[0]; oy = obs[1]; orad = obs[2]
    ship_p = load_ship_param

    if type == 0:    
        cbf = np.sqrt( (x-ox)**2 + (y - oy)**2) - orad

    if type == 1:    
        B = np.sqrt( (x-ox)**2 + (y - oy)**2) - orad
        Bdot = ((x-ox)*v*cos(psi) + (y-oy)*v*sin(psi))/np.sqrt((x-ox)**2 + (y - oy)**2)
        cbf = Bdot + ship_p.gamma1/orad*B

    if type == 2:
        R = v/ship_p.rmax*ship_p.gamma_TC1
        B1 = np.sqrt( (ox-x-R*cos(psi-np.pi/2))**2 + (oy-y-R*sin(psi-np.pi/2))**2) - (orad+R)
        B2 = np.sqrt( (ox-x-R*  cos(psi+np.pi/2))**2 + (oy-y-R*sin(psi+np.pi/2))**2) - (orad+R)
        if ship_p.TCCBF == 3:
            cbf = np.log((np.exp(B1)+np.exp(B2)-1))
        if ship_p.TCCBF == 2:
            cbf = B2
        if ship_p.TCCBF == 1:
            cbf = B1

    if type == 3:
        R = v/ship_p.rmax*ship_p.gamma_TC1
        B1 = np.sqrt( (ox-x-R*cos(psi-np.pi/2))**2 + (oy-y-R*sin(psi-np.pi/2))**2) - (orad+R)
        B2 = np.sqrt( (ox-x-R*  cos(psi+np.pi/2))**2 + (oy-y-R*sin(psi+np.pi/2))**2) - (orad+R)
        cbf = B2

    return cbf

if __name__ == '__main__':
    main(2,'overtaking',10)
    main(2,'single_static_straight',10)
    main(2,'static_narrow',10)
    main(2,'static_straight',10)
    gamma = 1.0
    main(1,'overtaking',10,gamma)
    main(1,'single_static_straight',10,gamma)
    main(1,'static_narrow',10,gamma)
    main(1,'static_straight',10,gamma)
    