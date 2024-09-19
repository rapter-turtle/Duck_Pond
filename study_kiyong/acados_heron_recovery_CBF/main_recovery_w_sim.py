from plot_asv import *
from gen_ref import *
from acados_setting import *
from recovery_simulator import*
from l_adaptive import*
import numpy as np

def main():

    Fmax = 60
    N_horizon = 10
    con_dt = 0.2
    Tf = int(N_horizon*con_dt)
    T_final = 200
    simulation_dt = 0.1

    x_tship = np.array([10.0, 10.0, 0.1, 0]) # x,y,psi,u

    x0 = np.array([-3.0, 4.0, 0.0, 0, 0, 0, 0, 0])

    head_dist = 1.0
    head_point = x0
    head_point[0] = x0[0] + head_dist*np.cos(x0[2])
    head_point[1] = x0[1] + head_dist*np.sin(x0[2])
    ocp_solver, integrator = setup_recovery(head_point, Fmax, N_horizon, Tf)

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu
    ncbf = 1

    simX = np.zeros((int(T_final/simulation_dt)+1, nx))
    simX_tship = np.zeros((int(T_final/simulation_dt)+1, 4))
    simU = np.zeros((int(T_final/simulation_dt)+1, nu))
    simX[0,:] = x0
    simX_tship[0,:] = x_tship
    CBF = np.zeros((int(T_final/simulation_dt)+1, ncbf*2))
    k = 0

    mpc_pred_list = []

    t_preparation = np.zeros((int(T_final/con_dt)))
    t_feedback = np.zeros((int(T_final/con_dt)))

    param_estim = np.zeros((3))
    param_filtered = np.zeros((2))
    state_estim = np.array([0.0,0.0, 0.0])


    l1_control = np.zeros((2))
    DOB = np.zeros((2))
    con = np.zeros((3))
    disturbance_state = np.zeros((6))
    disturbance_u = 0.0
    disturbance_v = 0.0
    disturbance_r = 0.0

    dob_save = np.zeros((int(T_final/simulation_dt)+1, 10))

    extra_control = np.zeros((2))
    
    # do some initial iterations to start with a good initial guess
    num_iter_initial = 5
    for _ in range(num_iter_initial):
        ocp_solver.solve_for_x0(x0_bar = head_point)

    # closed loop
    for i in range(int(T_final/simulation_dt)):
        if i%int(con_dt/simulation_dt) == 0:
            if i != 0:
                k += 1

            print(i)

            v_x_tship = x_tship[3]*np.cos(x_tship[2])
            v_y_tship = x_tship[3]*np.sin(x_tship[2])

            # for j in range(N_horizon):
            #     yref = np.array([x_tship[0]+v_x_tship*j*con_dt, 
            #                     x_tship[1]+v_y_tship*j*con_dt,
            #                     x_tship[2], 0, 0, 0, 0, 0, 0, 0])
            #     ocp_solver.cost_set(j, "yref", yref)


            # yref_N = np.array([x_tship[0]+v_x_tship*N_horizon*con_dt, 
            #                 x_tship[1]+v_y_tship*N_horizon*con_dt,
            #                 x_tship[2], 0, 0, 0, 0, 0])
            # ocp_solver.cost_set(N_horizon, "yref", yref_N)

            for j in range(N_horizon):
                yref = np.array([x_tship[0], 
                                x_tship[1],
                                x_tship[2], 0, 0, 0, 0, 0, 0, 0])
                                
                ocp_solver.cost_set(j, "yref", yref)


            yref_N = np.array([x_tship[0], 
                            x_tship[1],
                            x_tship[2], 0, 0, 0, 0, 0])
            ocp_solver.cost_set(N_horizon, "yref", yref_N)

            dist = 2.5
            oa = np.tan(x_tship[2]) 
            ob = -1 
            oc = (x_tship[1]-dist*np.cos(x_tship[2])) - (x_tship[0]+dist*np.sin(x_tship[2]))*np.tan(x_tship[2]) 
            con_pos = np.array([param_estim[0], param_estim[1], param_estim[2], disturbance_u, disturbance_r])
            # con_pos = np.array([0.0, 0.0,0.0, disturbance_u, disturbance_r])
            for j in range(N_horizon):
                ocp_solver.set(j, "p", con_pos)
            ocp_solver.set(N_horizon, "p", con_pos)
            con_pos = np.array([oa, ob, oc, disturbance_u, disturbance_r])
            # print(con_pos)

            # preparation phase
            ocp_solver.options_set('rti_phase', 1)
            status = ocp_solver.solve()
            t_preparation[k] = ocp_solver.get_stats('time_tot')

            
            # set initial state
            # CG = simX[i, :]
 
            ocp_solver.set(0, "lbx", simX[i,:])
            ocp_solver.set(0, "ubx", simX[i,:])

            # feedback phase
            ocp_solver.options_set('rti_phase', 2)
            status = ocp_solver.solve()
            t_feedback[k] = ocp_solver.get_stats('time_tot')
            # print(ocp_solver.get(0, "p")[0:3])
            # print(simU[k,:])
            
            print(t_preparation[k] + t_feedback[k])

            mpc_pred = []
            for j in range(N_horizon+1):
                mpc_pred.append(ocp_solver.get(j, "x")[0:2]) 
            mpc_pred_array = np.vstack(mpc_pred)
            mpc_pred_list.append(mpc_pred_array)


        ##### L1 adaptive #####   
        state_estim, param_estim, thrust = DOB_CBF(simX[i, :], state_estim, simulation_dt, param_estim, x_tship)
        # extra_control = np.array([0.0,0.0])
        # extra_control = L1_thruster
        # ##### L1 adaptive #####



        #################################### Disturbance ####################################
        wind_direction = 180*3.141592/180
        wind_speed = 4.0
        psi = simX[i,2]
        un = simX[i,3]
        vn = simX[i,4]

        ## Wave Disturbance ## 
        # wave_disturbance(disturbance_state, wave_direction, wind_speed, omega, lamda, Kw, sigmaF1, sigmaF2, dt):
        disturbance_state[:3], XY_wave_force = wave_disturbance(disturbance_state[:3], wind_direction, wind_speed, 0.1, 0.1, 0.1, 1, 0.1, simulation_dt)
        disturbance_state[3:6], N_wave_force = wave_disturbance(disturbance_state[3:6], wind_direction, wind_speed, 0.1, 0.1, 0.1, 0.5, 0.1, simulation_dt)
        
        X_wave_force = XY_wave_force*np.cos(wind_direction - psi)
        Y_wave_force = XY_wave_force*np.sin(wind_direction - psi)

        U_wave_force = X_wave_force*np.cos(psi) + Y_wave_force*np.sin(psi)
        V_wave_force = -X_wave_force*np.sin(psi) + Y_wave_force*np.cos(psi)
        
        ## Wind Disturbance ## 
        Afw = 0.3  #Frontal area
        Alw = 0.44  #Lateral area
        LOA = 1.3  #Length
        lau = 1.2  # air density
        CD_lAF = 0.55
        delta = 0.6
        CDl = CD_lAF*Afw/Alw
        CDt = 0.9

        u_rel_wind = un - wind_speed*np.cos(wind_direction - psi)
        v_rel_wind = vn - wind_speed*np.sin(wind_direction - psi)
        gamma = -np.arctan2(v_rel_wind, u_rel_wind)

        Cx = CD_lAF*np.cos(gamma)/(1 - delta*0.5*(1-CDl/CDt)*(np.sin(2*gamma))**2)
        Cy = CDt*np.sin(gamma)/(1 - delta*0.5*(1-CDl/CDt)*(np.sin(2*gamma))**2)
        Cn = -0.18*(gamma - np.pi*0.5)*Cy

        X_wind_force = 0.5*lau*(u_rel_wind**2 + v_rel_wind**2)*Cx*Afw
        Y_wind_force = 0.5*lau*(u_rel_wind**2 + v_rel_wind**2)*Cy*Alw
        N_wind_force = 0.5*lau*(u_rel_wind**2 + v_rel_wind**2)*Cn*Alw*LOA    

        U_wind_force = X_wind_force*np.cos(psi) + Y_wind_force*np.sin(psi)
        V_wind_force = -X_wind_force*np.sin(psi) + Y_wind_force*np.cos(psi)

        # U_wave_force = 0.0
        # V_wave_force = 0.0 
        # N_wave_force = 0.0

        # U_wind_force = 0.0   
        # N_wind_force = 3.0


        M = 37.758 
        I = 18.35
        head_dist = 1.0        
        disturbance = np.array([U_wave_force + U_wind_force, V_wave_force + V_wind_force, N_wave_force + N_wind_force])
        disturbance_head = np.array([disturbance[0]*np.cos(psi) - disturbance[1]*np.sin(psi) - disturbance[2]*head_dist*np.sin(psi), disturbance[0]*np.sin(psi) + disturbance[1]*np.cos(psi) + disturbance[2]*head_dist*np.cos(psi), 0.0])
        # disturbance = np.array([0.0,0.0,0.0])
        # print("Disturbance : ",[X_wind_force, Y_wind_force])
        # print("Estimated   : ",param_estim*M)
        # print("head point : ", [, Y_wind_force])
        ##########################################################################################
        
        
        # simulate system
        L = 0.6
        L1_XN = np.array([param_estim[0]*M, param_estim[1]*M,param_estim[2]*18.0 ])
        

        ##### CBF ####

        psi = simX[i,2]
        xn = simX[i,0]
        yn = simX[i,1]

        tpsi = -x_tship[2]
        xt = x_tship[0]
        yt = x_tship[1]
        
        beta = 7

        x_head = xn + head_dist*np.cos(psi)
        y_head = yn + head_dist*np.sin(psi)

        xr = (x_head - xt)*np.cos(tpsi) - (y_head - yt)*np.sin(tpsi)
        yr = (x_head - xt)*np.sin(tpsi) + (y_head - yt)*np.cos(tpsi)


        V = -(beta*yr*yr)/(1+yr*yr) - xr

  
        # print(xr, yr, V)
        if i == 0:
            V0 = V
        V_g = V0*np.exp(-0.1*simulation_dt*i)
        CBF[i,:] = np.hstack([V,V_g])


        # dob_save[i,:] = np.hstack((disturbance_head, L1_thruster, L1_XN, DOB))
        dob_save[i,:] = np.hstack((disturbance, extra_control, L1_XN, DOB))
        simU[i, :] = ocp_solver.get(0, "u")
        simX[i+1, :], x_tship = recover_simulator(simX[i, :], x_tship, simU[i,:], simulation_dt, disturbance ,thrust, param_estim )

        simX_tship[i+1, :] = x_tship


   

    simU[i+1, :] = simU[i, :]
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
    plot_iter = 5
    dt_gap = int(con_dt/simulation_dt)
    t = np.arange(0, T_final, simulation_dt)

    animateASV_recovery(simX[::dt_gap*plot_iter,:], simU[::dt_gap*plot_iter,:], 
                        simX_tship[::dt_gap*plot_iter,:], mpc_pred_list[::plot_iter], 
                        con_pos, t[::dt_gap*plot_iter], Fmax, dob_save[::dt_gap*plot_iter], CBF[::dt_gap*plot_iter,:])
    
    ## plot_iter - animateASV_recovery 안에 포함시키는거로 바꾸기

    # np.save('disturbances.npy', disturbances)

if __name__ == '__main__':
    main()
