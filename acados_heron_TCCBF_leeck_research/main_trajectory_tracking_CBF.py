from plot_asv import animateASV, plot_inputs
from gen_ref import *
from acados_setting import *
from load_param import load_param

def main():
    heron_p = load_param

    Fmax = heron_p.Fmax
    N_horizon = heron_p.N
    dt = heron_p.dt
    Tf = int(dt*N_horizon)
    Nsim = 300
    x0 = np.array([0.0, 2.0, 0.0 , 0.9, 0.0, 15, 15])
    
    con_dt = dt
    ref_dt = 0.01
    ref_iter = int(con_dt/ref_dt)

    ref_type = 0

    if ref_type == 0:
        reference = generate_figure_eight_trajectory((Nsim+N_horizon*2)*con_dt, ref_dt, 12, 6, 10)
    if ref_type == 1:
        reference = generate_figure_eight_trajectory_con((Nsim+N_horizon*2)*con_dt, ref_dt, 15, 12, 15)    
    if ref_type == 2:
        wpt_L = 10
        x0 = np.array([-wpt_L, -wpt_L, 0.0 , 1.0, 0.0, 15, 15])
        waypoints = [(-wpt_L, -wpt_L), (wpt_L*3, -wpt_L), (wpt_L*3, wpt_L), (-wpt_L, wpt_L)]
        desired_speed = 1.0  # meters per second
        repeat_wpt = 0
        reference = generate_wpt_trajectory(waypoints, repeat_wpt, desired_speed, ref_dt)

    reference = reference[::ref_iter,:]
    ocp_solver, integrator = setup_trajectory_tracking(x0)

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
        if ref_type < 2:
            for j in range(N_horizon):
                yref = np.hstack((reference[i+j,:],0,0,0,0))
                ocp_solver.cost_set(j, "yref", yref)
            
            yref = np.hstack((reference[i:i + N_horizon+1, :], np.zeros((N_horizon+1, 2))))
            yref_list.append(yref)

            yref_N = np.hstack((reference[i+N_horizon,:],0,0))
            ocp_solver.cost_set(N_horizon, "yref", yref_N)

        ## WPT following
        if ref_type == 2:
            ind = np.argmin(np.linalg.norm(reference[:,0:2]-simX[i, 0:2], axis=1))
            for j in range(N_horizon):
                yref = np.hstack((reference[ind+j,:],0,0,0,0))
                ocp_solver.cost_set(j, "yref", yref)

            yref = np.hstack((reference[ind:ind + N_horizon+1, :], np.zeros((N_horizon+1, 2))))
            yref_list.append(yref)

            yref_N = np.hstack((reference[ind+N_horizon,:],0,0))
            ocp_solver.cost_set(N_horizon, "yref", yref_N)


        
        ox1 = 10.0; oy1 = -5.5; or1 = 0.75
        ox2 = -5.0; oy2 = 2.5; or2 = 1.0
        ox3 = 16.0; oy3 = 0.0; or3 = 1.5
        ox4 = -13.0; oy4 = 5.0; or4 = 1.75
        ox5 = -6.0; oy5 = -5.5; or5 = 1.0
        
        ox1 =  10.5; oy1 =  0.0;  or1 = 1.5
        ox2 = -10.5; oy2 =  0.0;  or2 = 1.5
        ox3 =  0.0;  oy3 =  10.5; or3 = 1.5
        ox4 =  5.0;  oy4 = -10.5; or4 = 3.5
        ox5 =  0.0;  oy5 =  0.0;  or5 = 1.5
        
        # t1 = 1
        # t2 = 2
        # t3 = 3
        # t4 = 4
        # t5 = 5
        # ox1 = circle_A*np.sin(t1); oy1 = circle_A*np.cos(t1); or1 = 0.85
        # ox2 = circle_A*np.sin(t2); oy2 = circle_A*np.cos(t2); or2 = 1.25
        # ox3 = circle_A*np.sin(t3); oy3 = circle_A*np.cos(t3); or3 = 1.5
        # ox4 = circle_A*np.sin(t4); oy4 = circle_A*np.cos(t4); or4 = 1.25
        # ox5 = circle_A*np.sin(t5); oy5 = circle_A*np.cos(t5); or5 = 1.7
    
        obs_pos = np.array([ox1, oy1, or1, 
                            ox2, oy2, or2, 
                            ox3, oy3, or3, 
                            ox4, oy4, or4, 
                            ox5, oy5, or5])        
        obs_list.append(obs_pos)

        for j in range(N_horizon):
            ocp_solver.set(j, "p", obs_pos)
        ocp_solver.set(N_horizon, "p", obs_pos)


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
    plot_iter = 20
    # plot_iter = len(simU)-1

    animateASV(simX, simU, reference, yref_array,mpc_pred_list, obs_array, plot_iter)
    
    simU2 = simU**2
    simU1 = simX[:,5:7]**2
    udel_square_cost = np.sum(simU2)
    u_square_cost = np.sum(simU1)
    print('input delta cost:')
    print(udel_square_cost)
    print('input cost:')
    print(u_square_cost)
    
    t = np.arange(0, con_dt*Nsim, con_dt)
    plot_inputs(t, reference, simX, simU, Fmax)


if __name__ == '__main__':
    main()
