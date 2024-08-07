import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from load_param import load_kinematic_param
from matplotlib import cm, animation
from matplotlib.colors import Normalize
import time
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def animateASV(states, inputs, target_speed, mpc_result, obs_pos, cbf_and_dist, plot_iter, comptime, mode, obs_index,target_x):
    cbf_max = 20
    fig, axs = plt.subplots(3,4, figsize=(15,8))
    # Combine first two columns for the ASV plot
    fig.delaxes(axs[0, 1])
    fig.delaxes(axs[1, 0])
    fig.delaxes(axs[1, 1])
    ax_asv = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)  # Combine first row for ASV
    ax_asv_zoom = plt.subplot2grid((3, 4), (0, 2), colspan=1, rowspan=2)  # Combine first row for ASV
    # CBF animation setup
    ax_compt = axs[0, 3]
    # ax_cbf1 = axs[1, 0]
    # ax_cbf2 = axs[2, 1]
    ax_cbf3 = axs[1, 3]
    ax_surge = axs[2, 0]
    ax_rot = axs[2, 1]
    ax_acc = axs[2, 2]
    ax_del_con = axs[2, 3]
    

    vehicle_p = load_kinematic_param
    dt = vehicle_p.dt

    size = 1.5
    hullLength = 0.7 * size # Length of the hull
    hullWidth = 0.2 * size   # Width of each hull
    separation = 0.2 * size  # Distance between the two hulls
    bodyLength = hullLength  # Length of the body connecting the hulls
    bodyWidth = 0.25 * size  # Width of the body connecting the hulls

    # Define the grid for plotting
    x_range = np.linspace(states[0,0], states[-1,0], 200)
    y_range = np.linspace(-20, 20, 200)
    

    X, Y = np.meshgrid(x_range, y_range)
    
    FS = 16

    ## Create a scatter plot for the heatmap and initialize the colorbar
    # norm = Normalize(vmin=np.min(states[:, 3]), vmax=np.max(states[:, 3]))
    norm = Normalize(vmin=vehicle_p.vmin, vmax=vehicle_p.vmax)
    heatmap = ax_asv.scatter(states[:, 0], states[:, 1], c=states[:, 3], norm=norm, cmap=cm.rainbow, edgecolor='none', marker='o')
    cbar = plt.colorbar(heatmap, ax=ax_asv, fraction=0.025)
    cbar.set_label("velocity in [m/s]",fontsize=FS, labelpad=15)

    cbar_ax = fig.add_axes([0.08, 0.93, 0.35, 0.03])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap='bone', norm=mcolors.Normalize(vmin=-10, vmax=cbf_max))
    sm.set_array([])
    cbar_hk = fig.colorbar(sm, ax=ax_asv, cax=cbar_ax, orientation='horizontal')
    cbar_hk.ax.set_title("CBF values", fontsize=FS-2)
    cbar_hk.ax.axvline(x=(0 - (-50)) / (200 - (-50)), color='red', linewidth=2)  # Adjust the normalization accordingly

    # Create an inset axis for the zoom-in plot
    # ax_inset = inset_axes(ax_asv, width="50%", height="35%", loc='upper right')

    def update(frame):
        print(frame)
        position = states[frame,0:2]
        heading = states[frame,2]        
        theta = np.linspace( 0 , 2 * np.pi , 100 )

        # Clear the previous plot
        ax_asv.clear()
        ax_asv_zoom.clear()

        
        if (vehicle_p.CBF>0) and (vehicle_p.CBF_plot==1):
            hk = np.ones_like(X)*1000
            for i in range(obs_index):        
                # Initialize hk array
                # Calculate hk for each point in the grid
                for j in range(len(x_range)):
                    for k in range(len(y_range)):
                        x = X[j, k]
                        y = Y[j, k]
                        head_ang = states[frame,2]  # Assume theta = 0 for simplicity
                        u = states[frame,3]  # Assume constant speed for simplicity

                        if (vehicle_p.CBF==1):
                            B = np.sqrt((x - obs_pos[frame][5*i+0]) ** 2 + (y - obs_pos[frame][5*i+1]) ** 2) - obs_pos[frame][5*i+2]
                            Bdot = ((x - obs_pos[frame][5*i+0]) * (u * np.cos(head_ang)) + (y - obs_pos[frame][5*i+1]) *(u * np.sin(head_ang))) / np.sqrt((x - obs_pos[frame][5*i+0]) ** 2 + (y - obs_pos[frame][5*i+1]) ** 2)
                            # hk[j, k] = np.min((hk[j, k], Bdot + vehicle_p.gamma1/obs_pos[frame][5*i+2] * B))
                            hk[j, k] = np.min((hk[j, k], Bdot + vehicle_p.gamma1 * B))
                        elif (vehicle_p.CBF==2):
                            R = u/vehicle_p.rmax*vehicle_p.gamma_TC1
                            B1 = np.sqrt( (obs_pos[frame][5*i+0]-x-R*np.cos(head_ang-np.pi/2))**2 + (obs_pos[frame][5*i+1]-y-R*np.sin(head_ang-np.pi/2))**2) - (obs_pos[frame][5*i+2]+R)
                            B2 = np.sqrt( (obs_pos[frame][5*i+0]-x-R*np.cos(head_ang+np.pi/2))**2 + (obs_pos[frame][5*i+1]-y-R*np.sin(head_ang+np.pi/2))**2) - (obs_pos[frame][5*i+2]+R)
                            if vehicle_p.TCCBF == 1:
                                hk[j, k] = np.min((hk[j, k], B1))
                            if vehicle_p.TCCBF == 2:
                                hk[j, k] = np.min((hk[j, k], B2))
                            if vehicle_p.TCCBF == 3:
                                hk[j, k] = np.min((hk[j, k], np.max((B1,B2))))

                        if np.abs(np.arctan2((obs_pos[frame][5*i+1]-y),(obs_pos[frame][5*i+0]-x))-head_ang)>np.pi/2:
                            hk[j, k] = cbf_max
            # Plotting
            if vehicle_p.CBF==2 or vehicle_p.CBF==1 or vehicle_p.CBF==3:  # Only add the colorbar once
                ax_asv.contourf(X, Y, hk, levels=np.linspace(hk.min(),hk.max(),15), alpha=0.85, cmap=cm.bone)
                ax_asv.contourf(X, Y, hk, levels=[-0.02, 0.02], colors=['red'], alpha=1)
                ax_asv.contourf(X, Y, hk, levels=[-0.02+2, 0.02+2], alpha=1)
                ax_asv.contourf(X, Y, hk, levels=[-0.02+4, 0.02+4], alpha=1)
                ax_asv.contourf(X, Y, hk, levels=[-0.02+6, 0.02+6], alpha=1)
                ax_asv.contourf(X, Y, hk, levels=[-0.02+8, 0.02+8], alpha=1)
                ax_asv.contourf(X, Y, hk, levels=[-0.02+10, 0.02+10], alpha=1)

        radius = vehicle_p.radius
        a = position[0] + radius * np.cos( theta )
        b = position[1] + radius * np.sin( theta )    
        ax_asv.fill(a, b, color='green', alpha=0.1)
        
        # Define the vertices of the two hulls
        hull1 = np.array([[-hullLength/2, hullLength/2, hullLength/2, -hullLength/2, -hullLength/2, -hullLength/2],
                        [hullWidth/2, hullWidth/2, -hullWidth/2, -hullWidth/2, 0, hullWidth/2]])

        hull2 = np.array([[-hullLength/2, hullLength/2, hullLength/2, -hullLength/2, -hullLength/2, -hullLength/2],
                        [hullWidth/2, hullWidth/2, -hullWidth/2, -hullWidth/2, 0, hullWidth/2]])

        # Define the vertices of the body connecting the hulls
        body = np.array([[-bodyWidth/2, bodyWidth/2, bodyWidth/2, -bodyWidth/2, -bodyWidth/2],
                        [(separation-hullWidth)/2, (separation-hullWidth)/2, -(separation-hullWidth)/2, -(separation-hullWidth)/2, (separation-hullWidth)/2]])

        # Combine hulls into a single structure
        hull2[1, :] = hull2[1, :] - separation/2
        hull1[1, :] = hull1[1, :] + separation/2

        # Rotation matrix for the heading
        R = np.array([[np.cos(heading), -np.sin(heading)],
                    [np.sin(heading), np.cos(heading)]])

        # Rotate the hulls and body
        hull1 = R @ hull1
        hull2 = R @ hull2
        body = R @ body

        # Translate the hulls and body to the specified position
        hull1 = hull1 + np.array(position).reshape(2, 1)
        hull2 = hull2 + np.array(position).reshape(2, 1)
        body = body + np.array(position).reshape(2, 1)

        # Calculate the direction vector for the heading
        arrow_length = 0.5
        direction = np.array([np.cos(heading), np.sin(heading)]) * arrow_length

        # Plot the ASV
        ax_asv.fill(hull1[0, :], hull1[1, :], 'b', alpha=0.96)
        ax_asv.fill(hull2[0, :], hull2[1, :], 'b', alpha=0.96)
        ax_asv.fill(body[0, :], body[1, :], 'b', alpha=0.96)
        
        ax_asv.arrow(position[0], position[1], direction[0], direction[1], head_width=0.1, head_length=0.1, fc='k', ec='k')       
        ax_asv.scatter(states[0:frame,0], states[0:frame,1], c=states[0:frame,3], norm=norm, cmap=cm.rainbow, edgecolor='none', marker='.',linewidths=4)
        # ax_asv.plot(states[0:frame,0], states[0:frame,1], 'k', linewidth=3) 
        ax_asv.plot([states[0,0], states[-1,0]], [0, 0], 'k--')
        # ax_asv.plot(position[0], position[1], 'go', linewidth=2)   # Mark the position with a red dot


        a = position[0] + vehicle_p.radius * np.cos( theta )
        b = position[1] + vehicle_p.radius * np.sin( theta )    
        ax_asv.fill(a, b, color='green', alpha=(0.2))


        # ax_asv.plot(mpc_result[frame][:,0], mpc_result[frame][:,1], 'm--', linewidth=3) 
        ax_asv.xaxis.label.set_size(FS)
        ax_asv.yaxis.label.set_size(FS)

        # Update the inset axis
        zoom_size = 1  # Size of the zoomed-in area
        ax_asv_zoom.set_xlim(position[0] - 2*zoom_size, position[0] + 2*zoom_size)
        ax_asv_zoom.set_ylim(position[1] - 1.5*zoom_size, position[1] + 1.5*zoom_size)
        ax_asv_zoom.scatter(states[0:frame, 0], states[0:frame, 1], c=states[0:frame, 3], norm=norm, cmap=cm.rainbow, edgecolor='none', marker='o')
        ax_asv_zoom.fill(hull1[0, :], hull1[1, :], 'b', alpha=0.35)
        ax_asv_zoom.fill(hull2[0, :], hull2[1, :], 'b', alpha=0.35)
        ax_asv_zoom.fill(body[0, :], body[1, :], 'b', alpha=0.3)
        ax_asv_zoom.arrow(position[0], position[1], direction[0]*2, direction[1]*2, head_width=0.2, head_length=0.2, linewidth=2, fc='k', ec='k')
        ax_asv_zoom.plot([states[0, 0], states[-1, 0]], [0, 0], 'b--')
        # ax_inset.plot(position[0], position[1], 'go', linewidth=2)
        ax_asv_zoom.set_aspect('equal')

        # Plot the force vectors
        # ax_asv_zoom.arrow(force_left_position[0], force_left_position[1], force_left*np.cos(heading), force_left*np.sin(heading), head_width=0.2, head_length=0.2, linewidth=1, fc='r', ec='r')
        # ax_asv_zoom.arrow(force_right_position[0], force_right_position[1], force_right*np.cos(heading), force_right*np.sin(heading), head_width=0.2, head_length=0.2, linewidth=1, fc='g', ec='g')

        for jjj in range(0, vehicle_p.N):
            for i in range(5):
                # i = 4
                radius = obs_pos[frame][5*i+2] - vehicle_p.radius
                a = obs_pos[frame][5*i+0]+dt*jjj*obs_pos[frame][5*i+3] + radius * np.cos( theta )
                b = obs_pos[frame][5*i+1]+dt*jjj*obs_pos[frame][5*i+4] + radius * np.sin( theta )    
                if jjj == 0:
                    ax_asv.fill(a, b, facecolor='white', hatch='///', edgecolor='black')
                    ax_asv_zoom.fill(a, b, facecolor='white', hatch='///', edgecolor='black')
                else:
                    # ax_asv.fill(a, b, facecolor='white', hatch='//', edgecolor='black')
                    continue

        ax_asv_zoom.grid(True)

        ax_asv.set_aspect('equal')
        if mode == 'avoid':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-15,15))
            ax_asv.set(xlim=(0, target_x),ylim=(-10,10))
        elif mode == 'overtaking':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-10,10))
            ax_asv.set(xlim=(0, target_x),ylim=(-10,10))
        elif mode == 'static_narrow':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-6,6))
            ax_asv.set(xlim=(0, target_x),ylim=(-7.5,7.5))
        elif mode == 'static_straight':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-10,10))
            ax_asv.set(xlim=(0, target_x),ylim=(-10,10))
        elif mode == 'single_static_straight':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-10,10))
            ax_asv.set(xlim=(0, target_x),ylim=(-10,10))
        # ax_asv.grid(True)

        ax_asv.set_xlabel("x [m]", fontsize=FS)
        ax_asv.set_ylabel("y [m]", fontsize=FS)

        times = np.linspace(0, dt*frame, frame)
        # ax_cbf1.clear()
        # ax_cbf2.clear()
        ax_cbf3.clear()
        ax_compt.clear()
        ax_del_con.clear()
        ax_surge.clear()
        ax_rot.clear()
        ax_acc.clear()

        ax_compt.grid(True)
        for j in range(obs_index):
            ax_compt.plot(times,cbf_and_dist[0:frame,j])
        ax_compt.plot([0, len(cbf_and_dist)*dt],[0,0],'r--')
        ax_compt.set_xlim([0, len(cbf_and_dist)*dt])
        ax_compt.set_ylim([-1, cbf_max])
        ax_compt.set_xlabel("Time", fontsize=FS)
        ax_compt.set_ylabel("CBF", fontsize=FS)
        
        ax_cbf3.grid(True)
        for j in range(obs_index):            
            ax_cbf3.plot(times,cbf_and_dist[0:frame,5+j])
        ax_cbf3.plot([0, len(cbf_and_dist)*dt],[0,0],'r--')
        ax_cbf3.set_xlim([0, len(cbf_and_dist)*dt])
        # ax_cbf3.set_ylim([-5, np.max(cbf_and_dist[:,0])])
        ax_cbf3.set_ylim([-1, 10])
        ax_cbf3.set_xlabel("Time", fontsize=FS)
        ax_cbf3.set_ylabel("Closest Distance", fontsize=FS)

        ax_cbf3.xaxis.label.set_size(FS)
        ax_cbf3.yaxis.label.set_size(FS)

        ax_compt.xaxis.label.set_size(FS)
        ax_compt.yaxis.label.set_size(FS)

        t = np.arange(0, dt*frame, dt)
        ax_surge.plot(t, states[0:frame,3], 'k', linewidth=2) 
        ax_surge.plot(t, states[0:frame,3]*0 + target_speed, 'b--', linewidth=2) 
        ax_surge.set_xlabel("Time", fontsize=FS)
        ax_surge.set_ylabel("Surge Speed [m/s]", fontsize=FS)
        ax_surge.grid(True)
        ax_surge.autoscale(enable=True, axis='x', tight=True)
        ax_surge.autoscale(enable=True, axis='y', tight=True)
        ax_surge.set_ylim(vehicle_p.vmin, vehicle_p.vmax)

        # Plot the headings
        ax_rot.plot(t, states[0:frame,4], 'k', linewidth=2) 
        ax_rot.plot(t, states[0:frame,4]*0 - vehicle_p.rmax, 'r--')
        ax_rot.plot(t, states[0:frame,4]*0 + vehicle_p.rmax, 'r--')
        ax_rot.set_xlabel("Time", fontsize=FS)
        ax_rot.set_ylabel("Rot. Speed [rad/s]", fontsize=FS)
        ax_rot.grid(True)
        ax_rot.autoscale(enable=True, axis='x', tight=True)
        ax_rot.autoscale(enable=True, axis='y', tight=True)
        ax_rot.set_ylim(-vehicle_p.rmax-0.01, vehicle_p.rmax+0.01)

        ax_acc.plot(t, states[0:frame,5], 'k', linewidth=2) 
        ax_acc.plot(t, states[0:frame,5]*0 + vehicle_p.accmin, 'r--')
        ax_acc.plot(t, states[0:frame,5]*0 + vehicle_p.accmax, 'r--')
        ax_acc.set_xlabel("Time", fontsize=FS)
        ax_acc.set_ylabel("Acc [m/s^2]", fontsize=FS)
        ax_acc.grid(True)
        ax_acc.autoscale(enable=True, axis='x', tight=True)
        ax_acc.autoscale(enable=True, axis='y', tight=True)
        ax_acc.set_ylim(vehicle_p.accmin-0.01, vehicle_p.accmax+0.01)
        
        ax_del_con.plot(t, inputs[0:frame,0],  linewidth=2,label='drot') 
        ax_del_con.plot(t, inputs[0:frame,1],  linewidth=2,label='dacc') 
        ax_del_con.plot(t, inputs[0:frame,0]*0 - vehicle_p.daccmax, 'r--')
        ax_del_con.plot(t, inputs[0:frame,0]*0 + vehicle_p.daccmax, 'r--')
        ax_del_con.set_xlabel("Time", fontsize=FS)
        ax_del_con.set_ylabel("Del. Con", fontsize=FS)
        ax_del_con.grid(True)
        ax_del_con.legend()
        ax_del_con.autoscale(enable=True, axis='x', tight=True)
        ax_del_con.autoscale(enable=True, axis='y', tight=True)
        ax_del_con.set_ylim(-vehicle_p.daccmax-0.01, vehicle_p.daccmax+0.01)
        fig.tight_layout()  # axes 사이 간격을 적당히 벌려줍니다.

    frames = range(0, len(states), plot_iter)
    if plot_iter == len(states)-1:
        frames = range(len(states)-1, len(states))
    anim = FuncAnimation(fig, update, frames, init_function, repeat=False)
    # plt.show()
    # if vehicle_p.CBF == 2 or vehicle_p.CBF == 3:
    #     anim.save('Result_' + mode + '_cbf_type_' + str(vehicle_p.CBF) + '_TCCBF_type_=' + str(vehicle_p.TCCBF) + '_N=' + str(vehicle_p.N) + '.mp4', writer=animation.FFMpegWriter(fps=20))  
    # elif vehicle_p.CBF == 1:
    #     anim.save('Result_' + mode + '_cbf_type_' + str(vehicle_p.CBF) + '_EDCBF_gamma1_=' + str(0.75) + '_N=' + str(vehicle_p.N) + '.mp4', writer=animation.FFMpegWriter(fps=20))          
    # else:
    #     anim.save('Result_' + mode + 'cbf_type_' + str(vehicle_p.CBF) + 'N=' + str(vehicle_p.N) + '.mp4', writer=animation.FFMpegWriter(fps=20))  

    # if vehicle_p.CBF == 2 or vehicle_p.CBF == 3:
    #     anim.save('Result_' + mode + '_cbf_type_' + str(vehicle_p.CBF) + '_TCCBF_type_=' + str(vehicle_p.TCCBF) + '_N=' + str(10) + '.mp4', writer=animation.FFMpegWriter(fps=20))  
    # elif vehicle_p.CBF == 1:
    #     anim.save('Result_' + mode + '_cbf_type_' + str(vehicle_p.CBF) + '_EDCBF_gamma1_=' + str(0.75) + '_N=' + str(10) + '.mp4', writer=animation.FFMpegWriter(fps=20))          
    # else:
    #     anim.save('Result_' + mode + 'cbf_type_' + str(vehicle_p.CBF) + 'N=' + str(10) + '.mp4', writer=animation.FFMpegWriter(fps=20))  

    if vehicle_p.CBF == 2 or vehicle_p.CBF == 3:
        anim.save('Result_' + mode + '_TCCBF.mp4', writer=animation.FFMpegWriter(fps=20))  
    elif vehicle_p.CBF == 1:
        anim.save('Result_' + mode + '_EDCBF.mp4', writer=animation.FFMpegWriter(fps=20))          


    # if vehicle_p.CBF == 2 or vehicle_p.CBF == 3:
    #     anim.save('Result_' + mode + '_cbf_type_' + str(vehicle_p.CBF) + '_TCCBF_type_=' + str(vehicle_p.TCCBF) + '_N=' + str(vehicle_p.N) + '.gif', writer=animation.PillowWriter(fps=20))  
    # elif vehicle_p.CBF == 1:
    #     anim.save('Result_' + mode + '_cbf_type_' + str(vehicle_p.CBF) + '_EDCBF_gamma1_=' + str(vehicle_p.gamma1) + '_N=' + str(vehicle_p.N) + '.gif', writer=animation.PillowWriter(fps=20))          
    # else:
    #     anim.save('Result_' + mode + 'cbf_type_' + str(vehicle_p.CBF) + 'N=' + str(vehicle_p.N) + '.gif', writer=animation.PillowWriter(fps=20))  


def init_function():
    return

