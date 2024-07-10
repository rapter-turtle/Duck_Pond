from plot_ship import *
from acados_setting import *
from load_param import *

def main():
    ship_p = load_ship_param

    N_horizon = ship_p.N
    dt = ship_p.dt    
    con_dt = dt

    ship_p.CBF = 2
    mode = 'static_straight';   Tf = 350
    mode = 'static_narrow';   Tf = 100
    mode = 'avoid';           Tf = 70
    # mode = 'overtaking';      Tf = 150

    

    Nsim = int(Tf/dt)
    # Initial state
    target_speed = 2.5

    Fx_init = ship_p.Xu*target_speed + ship_p.Xuu*target_speed**2
    x0 = np.array([-50.0, # x
                   0.0, # y
                   0.0, # psi
                   target_speed, # vel
                   0.0, # rot-vel
                   Fx_init,  # Fx
                   0])  # Fn    

    ocp_solver, integrator = setup_wpt_tracking(x0)

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    simX = np.zeros((Nsim+1, nx))
    simU = np.zeros((Nsim+1, nu))
    simX[0,:] = x0
    yref_list = []
    obs_list = []
    mpc_pred_list = []

    t_preparation = np.zeros((Nsim))
    t_feedback = np.zeros((Nsim))

    # do some initial iterations to start with a good initial guess
    num_iter_initial = 5
    for _ in range(num_iter_initial):
        ocp_solver.solve_for_x0(x0_bar = x0)

    # closed loop
    for i in range(Nsim):
        print(i)
        for j in range(N_horizon):
            yref = np.hstack((0,0,0,target_speed,0,0,0,0,0))
            ocp_solver.cost_set(j, "yref", yref)
        yref_N = np.hstack((0,0,0,target_speed,0,0,0))
        ocp_solver.cost_set(N_horizon, "yref", yref_N)

        

        ## static obstacles - Straight
        if mode == 'static_straight':
            ox1 = 50; oy1 = +0.01; or1 = 10.0
            ox2 = 50; oy2 = +0.01; or2 = 10.0
            ox3 = 250; oy3 = +0.01; or3 = 25.0
            ox4 = 250; oy4 = +0.01; or4 = 25.0
            ox5 = 550; oy5 = +0.01; or5 = 50.0
            
            obs_pos = np.array([ox1, oy1, or1 + ship_p.radius, 0, 0,  
                                ox2, oy2, or2 + ship_p.radius, 0, 0,  
                                ox3, oy3, or3 + ship_p.radius, 0, 0,  
                                ox4, oy4, or4 + ship_p.radius, 0, 0,  
                                ox5, oy5, or5 + ship_p.radius, 0, 0])    
        
            obs_list.append(obs_pos)

            for j in range(N_horizon):
                ocp_solver.set(j, "p", obs_pos)
            ocp_solver.set(N_horizon, "p", obs_pos)

            
        ## Static obstacles - Narrow
        if mode == 'static_narrow':
            ox1 = 50; oy1 = +25; or1 = 20.0
            ox2 = 50; oy2 = -25; or2 = 20.0
            ox3 = 500; oy3 = +100; or3 = 0.001
            ox4 = 500; oy4 = +100; or4 = 0.001
            ox5 = 500; oy5 = +100; or5 = 0.001
            
            obs_pos = np.array([ox1, oy1, or1 + ship_p.radius, 0, 0,  
                                ox2, oy2, or2 + ship_p.radius, 0, 0,  
                                ox3, oy3, or3 + ship_p.radius, 0, 0,  
                                ox4, oy4, or4 + ship_p.radius, 0, 0,  
                                ox5, oy5, or5 + ship_p.radius, 0, 0])    
        
            obs_list.append(obs_pos)

            for j in range(N_horizon):
                ocp_solver.set(j, "p", obs_pos)
            ocp_solver.set(N_horizon, "p", obs_pos)

        ## Dynamic obstacles - Avoid              
        if mode == 'avoid':
            for j in range(N_horizon+1):
                ox1 = simX[i,0]-90
                oy1 = 0 
                or1 = 0.0
                ox2 = simX[i,0]-90
                oy2 = 0
                or2 = 0.0
                ox3 = simX[i,0]-90
                oy3 = 0
                or3 = 0.0
                ox4 = simX[i,0]-90
                oy4 = 0
                or4 = 0.0

                obs_speed = 2.0
                ox5 = 100 - (i+j)*dt*obs_speed;  
                oy5 = 0.001; 
                or5 = 10.0
                
                if ox5<-50:
                    ox5 = -50

                obs_pos = np.array([ox1, oy1, or1, 0, 0,  
                                    ox2, oy2, or2, 0, 0,  
                                    ox3, oy3, or3, 0, 0,  
                                    ox4, oy4, or4, 0, 0,  
                                    ox5, oy5, or5 + ship_p.radius, -obs_speed, 0])    
                if j == 0:
                    obs_list.append(obs_pos)

                # if simX[i, 0] > ox5:
                #     obs_pos = np.array([ox1, oy1, or1, 0, 0,  
                #                         ox2, oy2, or2, 0, 0,  
                #                         ox3, oy3, or3, 0, 0,  
                #                         ox4, oy4, or4, 0, 0,  
                #                         0, oy5, or5, -obs_speed, 0])    

                if j == N_horizon:
                    ocp_solver.set(N_horizon, "p", obs_pos)
                else:
                    ocp_solver.set(j, "p", obs_pos)         


        ## Dynamic obstacles - Overtaking              
        if mode == 'overtaking':
            for j in range(N_horizon+1):
                obs_speed = 1
                ox1 = 50 + (i+j)*dt*obs_speed; 
                oy1 = 0.001
                or1 = 0.0
                ox2 = 50 + (i+j)*dt*obs_speed; 
                oy2 = 0.001
                or2 = 0.0
                ox3 = 50 + (i+j)*dt*obs_speed; 
                oy3 = 0.001
                or3 = 0.0
                ox4 = 50 + (i+j)*dt*obs_speed; 
                oy4 = 0.001
                or4 = 0.0
                ox5 = 50 + (i+j)*dt*obs_speed; 
                oy5 = 0.001; 
                or5 = 10.0
                
                obs_pos = np.array([ox1, oy1, or1, 0, 0,  
                                    ox2, oy2, or2, 0, 0,  
                                    ox3, oy3, or3, 0, 0,  
                                    ox4, oy4, or4, 0, 0,  
                                    ox5, oy5, or5 + ship_p.radius, obs_speed, 0])    
                if j == 0:
                    obs_list.append(obs_pos)

                if j == N_horizon:
                    ocp_solver.set(N_horizon, "p", obs_pos)
                else:
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
        
        print((t_preparation[i] + t_feedback[i])*1000)

        mpc_pred = []
        for j in range(N_horizon+1):
            mpc_pred.append(ocp_solver.get(j, "x")[0:2]) 
        mpc_pred_array = np.vstack(mpc_pred)
        mpc_pred_list.append(mpc_pred_array)


        # simulate system
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])

        # print(simX[i, 3])
        # print(simU[i, :])
    simU[i+1, :] = simU[i, :]
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
    plot_iter = 20
    animateASV(simX, target_speed, mpc_pred_list, obs_array, plot_iter)

    t = np.arange(0, con_dt*Nsim, con_dt)
    plot_inputs(t, simX, simU, target_speed)


if __name__ == '__main__':
    main()
