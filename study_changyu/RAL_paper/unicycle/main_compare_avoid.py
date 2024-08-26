
from acados_setting import *
from load_param import *
from kinematic_integrator import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_param import load_kinematic_param
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors

def main(cbf_num,mode,prediction_horizon,gamma1, gamma2, target_speed):
    vehicle_p = load_kinematic_param
    vehicle_p.N = prediction_horizon

    N_horizon = vehicle_p.N
    dt = vehicle_p.dt    
    vehicle_p.CBF = cbf_num
    vehicle_p.gamma1 = gamma1
    vehicle_p.gamma2 = gamma2
    vehicle_p.gamma_TC2 = gamma2
    
    if mode == 'static':
        Tf = 25
        target_x = 50

    if mode == 'avoid':   
        Tf = 27
        target_x = 50
        
    if mode == 'overtaking':   
        Tf = 20
        target_x = 35

    Nsim = int(Tf/dt)
    # target_speed = 1.5
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

    for i in range(Nsim):
        # print(i)
        print('CBF Num:' + str(cbf_num) + ', N:' + str(prediction_horizon) + ', Mode:' + str(mode) + ', Iter:' + str(i+1) + '/'+ str(Nsim))
        
        for j in range(N_horizon):
            yref = np.hstack((0,0,0,target_speed,0,0,0,0))
            ocp_solver.cost_set(j, "yref", yref)
        yref_N = np.hstack((0,0,0,target_speed,0,0))
        ocp_solver.cost_set(N_horizon, "yref", yref_N)
        

        ## static obstacles - Straight
        if mode == 'static':
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

        ## Dynamic obstacles - Avoid              
        if mode == 'avoid':
            for j in range(N_horizon+1):                
                obs_speed = -0.75
                xinit = 30
                ox1 = xinit + (i+j)*dt*obs_speed;  
                oy1 = 0.001; 
                or1 = 5.0*obsrad_param
                ox2 = xinit + (i+j)*dt*obs_speed;  
                oy2 = 0.001; 
                or2 = 5.0*obsrad_param
                ox3 = xinit + (i+j)*dt*obs_speed;  
                oy3 = 0.001; 
                or3 = 5.0*obsrad_param
                ox4 = xinit + (i+j)*dt*obs_speed;  
                oy4 = 0.001; 
                or4 = 5.0*obsrad_param
                ox5 = xinit + (i+j)*dt*obs_speed;  
                oy5 = 0.001; 
                or5 = 5.0*obsrad_param
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
    time_save = t_preparation+t_feedback

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

    
    con_usage_all = vehicle_p.R[0,0]/2*drot_usage + vehicle_p.R[1,1]/2*dacc_usage + vehicle_p.Q[4,4]/2*rot_usage + vehicle_p.Q[5,5]/2*acc_usage
    
    return closest_dist, arrival_time , speed_var , con_usage_all , simX, simU, target_speed, obs_array, cbf_and_dist, time_save



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
        k = vehicle_p.TC_k
        B1 = np.sqrt( (ox-x-R*cos(psi-np.pi/2))**2 + (oy-y-R*sin(psi-np.pi/2))**2) - (orad+R)
        B2 = np.sqrt( (ox-x-R*cos(psi+np.pi/2))**2 + (oy-y-R*sin(psi+np.pi/2))**2) - (orad+R)
        cbf = 1/k*np.log((np.exp(k*B1)+np.exp(k*B2))/2)

    return cbf

if __name__ == '__main__':

    target_x = 50
    tf = 25
    
    
    g2 = 0.1
    g2_tc = 0.1
    mode = 'avoid'
    target_speed = 2
    # cd1, at1 ,sv1 ,con1, simX1, simU1, target_speed1, obs_array1, cbf_and_dist1, time_save1  = main(1,'static', 10, 0.15, 0.05, target_speed)
    cd2, at2 ,sv2 ,con2, simX2, simU2, target_speed2, obs_array2, cbf_and_dist2, time_save2  = main(1,'avoid', 10, 0.5, 0.05, target_speed)
    # cd3, at3 ,sv3 ,con3, simX3, simU3, target_speed3, obs_array3, cbf_and_dist3, time_save3  = main(1,'static', 10, 0.50, 0.05, target_speed)
    # cd4, at4 ,sv4 ,con4, simX4, simU4, target_speed4, obs_array4, cbf_and_dist4, time_save4  = main(1,'static', 10, 0.25, 0.01, target_speed)
    # cd5, at5 ,sv5 ,con5, simX5, simU5, target_speed5, obs_array5, cbf_and_dist5, time_save5  = main(1,'static', 10, 0.25, 0.10, target_speed)
    # cd6, at6 ,sv6 ,con6, simX6, simU6, target_speed6, obs_array6, cbf_and_dist6, time_save6  = main(2,'avoid', 10, 0.00, 0.01, target_speed)
    # cd7, at7 ,sv7 ,con7, simX7, simU7, target_speed7, obs_array7, cbf_and_dist7, time_save7  = main(2,'avoid', 10, 0.00, 0.02, target_speed)
    cd8, at8 ,sv8 ,con8, simX8, simU8, target_speed8, obs_array8, cbf_and_dist8, time_save8  = main(2,'avoid', 10, 1.00, 0.05, target_speed)
    # cd9, at9 ,sv9 ,con9, simX9, simU9, target_speed9, obs_array9, cbf_and_dist9, time_save9  = main(2,'avoid', 10, 0.00, 0.05, target_speed)
    
    # Combine data into a list of dictionaries
    sim_data = [
        # {"simX": simX1, "simU": simU1, "target_speed": target_speed1, "obs_array": obs_array1, "cbf_and_dist": cbf_and_dist1},
        {"simX": simX2, "simU": simU2, "target_speed": target_speed2, "obs_array": obs_array2, "cbf_and_dist": cbf_and_dist2},
        # {"simX": simX3, "simU": simU3, "target_speed": target_speed3, "obs_array": obs_array3, "cbf_and_dist": cbf_and_dist3},
        # {"simX": simX4, "simU": simU4, "target_speed": target_speed4, "obs_array": obs_array4, "cbf_and_dist": cbf_and_dist4},
        # {"simX": simX5, "simU": simU5, "target_speed": target_speed5, "obs_array": obs_array5, "cbf_and_dist": cbf_and_dist5},
        # {"simX": simX6, "simU": simU6, "target_speed": target_speed6, "obs_array": obs_array6, "cbf_and_dist": cbf_and_dist6},
        # {"simX": simX7, "simU": simU7, "target_speed": target_speed7, "obs_array": obs_array7, "cbf_and_dist": cbf_and_dist7},
        {"simX": simX8, "simU": simU8, "target_speed": target_speed8, "obs_array": obs_array8, "cbf_and_dist": cbf_and_dist8},
    ]
    plot_results = 0
    plot_box = 1
    if plot_results:

        colors = plt.cm.tab10(np.linspace(0, 1, 10))    
        fig, axs = plt.subplots(4,2, figsize=(11,10))
        # Combine first two columns for the ASV plot
        # fig.delaxes(axs[0, 2])
        ax_asv = plt.subplot2grid((4, 2), (0, 0), colspan=2, rowspan=2)  # Combine first row for ASV
        ax_compt = axs[2, 0]
        ax_cbf3 = axs[2, 1]
        ax_surge = axs[3, 0]
        ax_rot = axs[3, 1]
        # ax_acc = axs[1, 2]
            

        

        vehicle_p = load_kinematic_param
        dt = vehicle_p.dt

        size = 1
        hullLength = 0.8 * size # Length of the hull
        hullWidth = 0.35 * size   # Width of each hull
        separation = 0.2 * size  # Distance between the two hulls
        bodyLength = hullLength  # Length of the body connecting the hulls
        bodyWidth = 0.25 * size  # Width of the body connecting the hulls

        # Define the grid for plotting
        x_range = np.linspace(simX2[0,0], simX2[-1,0], 200)
        y_range = np.linspace(-10, 10, 200)
        

        X, Y = np.meshgrid(x_range, y_range)
        
        FS = 16

        ax_asv.clear()    
        for idx, data in enumerate(sim_data):
            simX = data["simX"]
            simU = data["simU"]
            target_speed = data["target_speed"]
            obs_array = data["obs_array"]
            cbf_and_dist = data["cbf_and_dist"]

            for i in range(len(simX)):
                if i%25 == 0:
                    position = simX[i,0:2]
                    heading = simX[i,2]
                    hull = np.array([[-hullLength/2, hullLength/2, hullLength/2+0.25, hullLength/2,-hullLength/2, -hullLength/2, -hullLength/2],
                                    [hullWidth/2, hullWidth/2, 0, -hullWidth/2, -hullWidth/2, 0, hullWidth/2]])

                    R = np.array([[np.cos(heading), -np.sin(heading)],
                                [np.sin(heading), np.cos(heading)]])

                    hull = R @ hull
                    hull = hull + np.array(position).reshape(2, 1)

                    # Calculate the direction vector for the heading
                    arrow_length = 1
                    direction = np.array([np.cos(heading), np.sin(heading)]) * arrow_length

                    # Plot the ASV
                    ax_asv.fill(hull[0, :], hull[1, :], alpha=0.4, color=colors[idx])
                    # ax_asv.arrow(position[0], position[1], direction[0], direction[1], head_width=0.1, head_length=0.1, fc='k', ec='k')                              
                    if idx == 1 and i > 0:
                        label = f'{i / 10:.1f} s'
                        if i < 250:
                            ax_asv.annotate(label, (position[0], position[1]), textcoords="offset points", xytext=(0, -20), ha='center', fontsize=FS-5, color='black')
                        else:
                            ax_asv.annotate(label, (position[0], position[1]), textcoords="offset points", xytext=(-10, 15), ha='center', fontsize=FS-5, color='black')
                        
            if idx == 0:
                lgd = 'MPC-EDCBF'
            elif idx == 1:             
                lgd = 'MPC-TCCBF'

            ax_asv.plot(simX[:,0], simX[:,1], linewidth=2, color=colors[idx],label=lgd)
        ax_asv.plot([simX[0,0], simX[-1,0]], [0, 0], 'k--')
        # Annotate specific times
        
        ax_asv.annotate('0.0 s', (obs_array[0][0], obs_array[0][1]), textcoords="offset points", xytext=(0, 20), ha='center', fontsize=FS-5, color='black')
        ax_asv.annotate('25.0 s', (obs_array[-1][0], obs_array[-1][1]), textcoords="offset points", xytext=(0, 20), ha='center', fontsize=FS-5, color='black')
        # Draw an arrow between the two annotated points
        start_position = obs_array[0][0:2] + [-1.5,1.75]
        end_position = obs_array[-1][0:2] + [1.5,1.75]
        print(start_position)
        ax_asv.annotate('', xy=end_position, xytext=start_position, arrowprops=dict(arrowstyle="->", color='black', lw=1.5))
        
        for i in range(len(simX)):
            if i%25 == 0:                    
                theta = np.linspace( 0 , 2 * np.pi , 100 )
                radius = obs_array[i][2]
                a = obs_array[i][0] + radius * np.cos( theta )
                b = obs_array[i][1] + radius * np.sin( theta )    
                ax_asv.fill(a, b, facecolor='k', alpha=0.1, edgecolor='gray')

        ax_asv.set_xlabel('x [m]', fontsize=FS)  # Set x-axis label and font size
        ax_asv.set_ylabel('y [m]', fontsize=FS)  # Set y-axis label and font size

        ax_asv.tick_params(axis='x', labelsize=FS)  # Set x-axis tick label size
        ax_asv.tick_params(axis='y', labelsize=FS)  # Set y-axis tick label size
        
        ax_asv.set_aspect('equal')
        ax_asv.set(xlim=(0, target_x),ylim=(-5,2.5))
        # ax_asv.grid(True)
        ax_asv.legend(fontsize=FS-1)


        times = np.linspace(0, dt*(len(simX)-1), len(simX))
        ax_cbf3.clear()
        ax_compt.clear()
        ax_surge.clear()
        ax_rot.clear()
        # ax_acc.clear()

        ax_compt.grid(True)
        j = 0
        
        for idx, data in enumerate(sim_data):
            cbf_and_dist = data["cbf_and_dist"]
            ax_compt.plot(times,cbf_and_dist[:,j], color=colors[idx])
        ax_compt.plot([0, len(cbf_and_dist)*dt],[0,0],'k--',linewidth=1)
        ax_compt.set_xlim([0, tf])
        ax_compt.set_ylim([-1, 25])
        ax_compt.set_xlabel("Time [s]", fontsize=FS)
        ax_compt.set_ylabel("CBF", fontsize=FS)
        ax_compt.tick_params(axis='x', labelsize=FS)  # Set x-axis tick label size
        ax_compt.tick_params(axis='y', labelsize=FS)  # Set y-axis tick label size
        
        ax_cbf3.grid(True)
        for idx, data in enumerate(sim_data):
            cbf_and_dist = data["cbf_and_dist"]
            ax_cbf3.plot(times,cbf_and_dist[:,5+j], color=colors[idx])
        ax_cbf3.plot([0, len(cbf_and_dist)*dt],[0,0],'k--',linewidth=1)
        ax_cbf3.set_xlim([0, tf])
        ax_cbf3.set_ylim([-1, 40])
        ax_cbf3.set_xlabel("Time [s]", fontsize=FS)
        ax_cbf3.set_ylabel("Closest Distance [m]", fontsize=FS)

        ax_cbf3.tick_params(axis='x', labelsize=FS)  # Set x-axis tick label size
        ax_cbf3.tick_params(axis='y', labelsize=FS)  # Set y-axis tick label size
        

        t = np.linspace(0, dt*(len(simX)-1), len(simX))
        
        ax_surge.plot(t, simX[:,3]*0 + target_speed, 'k--', linewidth=1) 
        for idx, data in enumerate(sim_data):
            simX = data["simX"]    
            ax_surge.plot(t, simX[:,3], linewidth=2, color=colors[idx]) 
        ax_surge.set_xlabel("Time [s]", fontsize=FS)
        ax_surge.set_ylabel(r"$u$ [m/s]", fontsize=FS)
        ax_surge.grid(True)
        ax_surge.autoscale(enable=True, axis='x', tight=True)
        ax_surge.autoscale(enable=True, axis='y', tight=True)
        ax_surge.set_xlim([0, tf])
        ax_surge.set_ylim(vehicle_p.vmin, vehicle_p.vmax)
        ax_surge.tick_params(axis='x', labelsize=FS)  # Set x-axis tick label size
        ax_surge.tick_params(axis='y', labelsize=FS)  # Set y-axis tick label size
        

        # Plot the headings
        for idx, data in enumerate(sim_data):
            simX = data["simX"]   
            ax_rot.plot(t, simX[:,4], linewidth=2, color=colors[idx]) 
        # ax_rot.plot(t, simX[:,4]*0 - vehicle_p.rmax, 'r--')
        # ax_rot.plot(t, simX[:,4]*0 + vehicle_p.rmax, 'r--')
        ax_rot.set_xlabel("Time [s]", fontsize=FS)
        ax_rot.set_ylabel(r"$r$ [rad/s]", fontsize=FS)
        ax_rot.grid(True)
        ax_rot.autoscale(enable=True, axis='x', tight=True)
        ax_rot.autoscale(enable=True, axis='y', tight=True)
        ax_rot.set_xlim([0, tf])
        ax_rot.set_ylim(-vehicle_p.rmax-0.005, vehicle_p.rmax+0.005)
        # ax_rot.set_ylim(-vehicle_p.rmax-0.0, vehicle_p.rmax+0.0)
        ax_rot.tick_params(axis='x', labelsize=FS)  # Set x-axis tick label size
        ax_rot.tick_params(axis='y', labelsize=FS)  # Set y-axis tick label size
        

        # for idx, data in enumerate(sim_data):
        #     simX = data["simX"]   
        #     ax_acc.plot(t, simX[:,5], linewidth=2, color=colors[idx])
        # # ax_acc.plot(t, simX[:,5]*0 + vehicle_p.accmin, 'r--')
        # # ax_acc.plot(t, simX[:,5]*0 + vehicle_p.accmax, 'r--')
        # ax_acc.set_xlabel("Time [s]", fontsize=FS)
        # ax_acc.set_ylabel(r"$a$ [$\rm m/s^2$]", fontsize=FS)
        # ax_acc.grid(True)
        # ax_acc.autoscale(enable=True, axis='x', tight=True)
        # ax_acc.autoscale(enable=True, axis='y', tight=True)
        # ax_acc.set_xlim([0, tf])
        # ax_acc.set_ylim(vehicle_p.accmin-0.01, vehicle_p.accmax+0.01)
        # ax_acc.set_ylim(vehicle_p.accmin-0.0, vehicle_p.accmax+0.0)
        # ax_acc.tick_params(axis='x', labelsize=FS)  # Set x-axis tick label size
        # ax_acc.tick_params(axis='y', labelsize=FS)  # Set y-axis tick label size
        
        
        fig.tight_layout()  # axes 사이 간격을 적당히 벌려줍니다.

        plt.show()
        
    
    # plt.pause(0.01)
    # plt.close('all')
    
    
    if plot_box:
        # # Sample data based on the provided outputs for each case
        data = {
            "Case": [
                "1 - avoid", "2 - avoid",
            ],
            "closest_dist": [cd2,cd8],
            "arrival_time":[ at2, at8],
            "speed_var":  [sv2,   sv8],
            "con_usage":  [con2,   con8],
        }

        # Create DataFrame
        df = pd.DataFrame(data)
        df = df.round(6)

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
        