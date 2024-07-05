from plot_asv import *
from gen_ref import *
from acados_setting import *
from recovery_simulator import*
from l_adaptive import*
from DOB import*
import numpy as np

def main():

    Fmax = 45
    Tf = 1
    N_horizon = 20
    Nsim = 1000
    simulation_dt = 0.01

    con_dt = (Tf/N_horizon)
    ref_dt = 0.01

    x_tship = np.array([10.0, 10.0, 0.1, 1]) # x,y,psi,u

    x0 = np.array([0.0, 2.0, 0.0 , 1, 0.0, 0, 0])
    ocp_solver, integrator = setup_recovery(x0, Fmax, N_horizon, Tf)

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    simX = np.zeros((int(Nsim*con_dt/simulation_dt)+1, nx))
    simX_tship = np.zeros((int(Nsim*con_dt/simulation_dt)+1, 4))
    simU = np.zeros((Nsim+1, nu))
    simX[0,:] = x0
    simX_tship[0,:] = x_tship
    k = 0

    mpc_pred_list = []

    t_preparation = np.zeros((Nsim))
    t_feedback = np.zeros((Nsim))

    param_estim = np.zeros((2))
    param_filtered = np.zeros((2))
    state_estim = np.array([0.0, 2.0, 0.0 , 1, 0.0])
    l1_control = np.zeros((2))
    extra_control = np.zeros((2))

    # do some initial iterations to start with a good initial guess
    num_iter_initial = 5
    for _ in range(num_iter_initial):
        ocp_solver.solve_for_x0(x0_bar = x0)

    # closed loop
    for i in range(int(Nsim*con_dt/simulation_dt)):
        if i%int(con_dt/simulation_dt) == 0:
            if i != 0:
                k += 1

            print(i)

            v_x_tship = x_tship[3]*np.cos(x_tship[2])
            v_y_tship = x_tship[3]*np.sin(x_tship[2])

            for j in range(N_horizon):
                yref = np.array([x_tship[0]+v_x_tship*j*con_dt, 
                                x_tship[1]+v_y_tship*j*con_dt,
                                x_tship[2], x_tship[3], 0, 0, 0, 0, 0])
                ocp_solver.cost_set(j, "yref", yref)


            yref_N = np.array([x_tship[0]+v_x_tship*N_horizon*con_dt, 
                            x_tship[1]+v_y_tship*N_horizon*con_dt,
                            x_tship[2], x_tship[3], 0, 0, 0])
            ocp_solver.cost_set(N_horizon, "yref", yref_N)


            dist = 2.5
            oa = np.tan(x_tship[2]) 
            ob = -1 
            oc = (x_tship[1]-dist*np.cos(x_tship[2])) - (x_tship[0]+dist*np.sin(x_tship[2]))*np.tan(x_tship[2]) 
            con_pos = np.array([oa, ob, oc])
            for j in range(N_horizon):
                ocp_solver.set(j, "p", con_pos)
            ocp_solver.set(N_horizon, "p", con_pos)

            # preparation phase
            ocp_solver.options_set('rti_phase', 1)
            status = ocp_solver.solve()
            t_preparation[k] = ocp_solver.get_stats('time_tot')

            # set initial state
            ocp_solver.set(0, "lbx", simX[i, :])
            ocp_solver.set(0, "ubx", simX[i, :])

            # feedback phase
            ocp_solver.options_set('rti_phase', 2)
            status = ocp_solver.solve()
            t_feedback[k] = ocp_solver.get_stats('time_tot')

            simU[k, :] = ocp_solver.get(0, "u")
            # print(simU[k,:])
            
            print(t_preparation[k] + t_feedback[k])

            mpc_pred = []
            for j in range(N_horizon+1):
                mpc_pred.append(ocp_solver.get(j, "x")[0:2]) 
            mpc_pred_array = np.vstack(mpc_pred)
            mpc_pred_list.append(mpc_pred_array)

        ##### L1 adaptive #####
        state_estim, param_estim, param_filtered = L1_control(simX[i, :], state_estim, param_filtered, simulation_dt, param_estim)
        #L1 control allocation
        M = 36 # Mass [kg]
        I = 8.35 # Inertial tensor [kg m^2]
        L = 0.73 # length [m]
        l1_control[0] = (M*param_filtered[0] - 2*(I/L)*param_filtered[1])*0.5
        l1_control[1] = (M*param_filtered[0] + 2*(I/L)*param_filtered[1])*0.5
        extra_control = l1_control
        # print(param_estim)
        print(param_filtered)
        ##### L1 adaptive #####
        

        ##### DOB #####
        # state_estim, param_estim = Disturbance_observer(simX[i, :], state_estim, simulation_dt, param_estim)
        # print(param_estim)
        ##### DOB #####



        # simulate system
        simX[i+1, :], x_tship = recover_simulator(simX[i, :], x_tship, simU[k,:], simulation_dt, i*simulation_dt, extra_control )

        simX_tship[i+1, :] = x_tship


    simU[k+1, :] = simU[k, :]
    mpc_pred_list.append(mpc_pred_array)

    # evaluate timings
    # scale to milliseconds
    t_preparation *= 1000
    t_feedback *= 1000
    print(f'Computation time in preparation phase in ms: \
            min {np.min(t_preparation):.3f} median {np.median(t_preparation):.3f} max {np.max(t_preparation):.3f}')
    print(f'Computation time in feedback phase in ms:    \
            min {np.min(t_feedback):.3f} median {np.median(t_feedback):.3f} max {np.max(t_feedback):.3f}')


    ocp_solver = None
    scale = 10
    plot_iter = int(con_dt/simulation_dt)

    animateASV_recovery(simX[::scale*plot_iter,:], simU[::scale,:], simX_tship[::scale*plot_iter,:], mpc_pred_list[::scale], con_pos)

    t = np.arange(0, con_dt*Nsim, con_dt)
    plot_inputs_recovery(t, simX[::plot_iter,:], simU, Fmax)


if __name__ == '__main__':
    main()
