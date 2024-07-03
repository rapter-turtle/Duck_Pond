from plot_asv import animateASV, plot_inputs
from gen_ref import *
from acados_setting import *

def main():

    Fmax = 45
    Tf = 4
    N_horizon = 20
    Nsim = 1000

    con_dt = (Tf/N_horizon)
    ref_dt = 0.01
    ref_iter = int(con_dt/ref_dt)

    reference = generate_figure_eight_trajectory((Nsim+N_horizon*2)*con_dt, ref_dt)
    # reference = generate_figure_eight_trajectory_con(100, ref_dt)
    reference = reference[::ref_iter,:]

    x0 = np.array([0.0, 2.0, 0.0 , 0.5, 0.0, 0, 0])
    ocp_solver, integrator = setup_trajectory_tracking(x0, Fmax, N_horizon, Tf)

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
            yref = np.hstack((reference[i+j,:],0,0,0,0))
            ocp_solver.cost_set(j, "yref", yref)
        
        yref = np.hstack((reference[i:i + N_horizon+1, :], np.zeros((N_horizon+1, 2))))
        yref_list.append(yref)

        yref_N = np.hstack((reference[i+N_horizon,:],0,0))
        ocp_solver.cost_set(N_horizon, "yref", yref_N)

        rad = 1
        obs_pos = np.array([0.0, 0.1, 0.5, 
                            6.0, -1.0*np.sin(i/10), 0.6, 
                            3.0, 1.0, 0.4, 
                            -3.0, 1.0, 0.4,
                            -6.0, 1.0*np.sin(i/10), 0.5])
        obs_list.append(obs_pos)

        for j in range(N_horizon):
            obs_pos = np.array([0.0, 0.1, 0.5, 
                                6.0, -1.0*np.sin((i+j*con_dt)/10), 0.6, 
                                3.0, 1.0, 0.4, 
                                -3.0, 1.0, 0.4,
                                -6.0, 1.0*np.sin((i+j*con_dt)/10), 0.5])
            ocp_solver.set(j, "p", obs_pos)
        obs_pos = np.array([0.0, 0.1, 0.5, 
                            6.0, -1.0*np.sin((i+j*con_dt)/10), 0.6, 
                            3.0, 1.0, 0.4, 
                            -3.0, 1.0, 0.4,
                            -6.0, 1.0*np.sin((i+N_horizon*con_dt)/10), 0.5])
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
        
        print(t_preparation[i] + t_feedback[i])

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
    plot_iter = 3
    animateASV(simX[::plot_iter,:], simU[::plot_iter,:], reference, yref_array[::plot_iter],mpc_pred_list[::plot_iter], obs_array)

    t = np.arange(0, con_dt*Nsim, con_dt)
    plot_inputs(t, reference, simX, simU, Fmax)


if __name__ == '__main__':
    main()
