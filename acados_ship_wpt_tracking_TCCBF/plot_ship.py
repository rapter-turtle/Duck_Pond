import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from load_param import load_ship_param
from matplotlib import cm, animation
from matplotlib.colors import Normalize
import time
import matplotlib.colors as mcolors

def animateASV(states, inputs, target_speed, mpc_result, obs_pos, cbf_and_dist, plot_iter, comptime, mode, obs_index):
    
    fig, axs = plt.subplots(3,3, figsize=(12,8))
    # Combine first two columns for the ASV plot
    fig.delaxes(axs[0, 1])
    fig.delaxes(axs[1, 0])
    fig.delaxes(axs[1, 1])
    ax_asv = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)  # Combine first row for ASV
    # CBF animation setup
    ax_compt = axs[0, 2]
    # ax_cbf1 = axs[1, 0]
    # ax_cbf2 = axs[2, 1]
    ax_cbf3 = axs[1, 2]
    ax_speed = axs[2, 0]
    ax_rot = axs[2, 1]
    ax_thrust = axs[2, 2]


    ship_p = load_ship_param
    dt = ship_p.dt

    # Define the geometry of the twin-hull ASV
    separation = ship_p.L*0.3  # Distance between the two hulls
    bodyWidth = ship_p.L  # Width of the body connecting the hulls

    # Define the grid for plotting
    x_range = np.linspace(states[0,0], states[-1,0], 200)
    y_range = np.linspace(-150, 150, 200)
    if mode == 'crossing':
        y_range = np.linspace(-180, 180, 200)

    if mode == 'single_static_straight':
        y_range = np.linspace(-60, 60, 200)


    X, Y = np.meshgrid(x_range, y_range)
    
    FS = 16

    ## Create a scatter plot for the heatmap and initialize the colorbar
    # norm = Normalize(vmin=np.min(states[:, 3]), vmax=np.max(states[:, 3]))
    norm = Normalize(vmin=0.5, vmax=2.5)
    heatmap = ax_asv.scatter(states[:, 0], states[:, 1], c=states[:, 3], norm=norm, cmap=cm.rainbow, edgecolor='none', marker='o')
    cbar = plt.colorbar(heatmap, ax=ax_asv, fraction=0.03)
    cbar.set_label("velocity in [m/s]",fontsize=FS-2)

    cbar_ax = fig.add_axes([0.09, 0.94, 0.5, 0.03])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap='bone', norm=mcolors.Normalize(vmin=-10, vmax=30))
    sm.set_array([])
    cbar_hk = fig.colorbar(sm, ax=ax_asv, cax=cbar_ax, orientation='horizontal')
    cbar_hk.ax.set_title("CBF values", fontsize=FS-2)
    cbar_hk.ax.axvline(x=(0 - (-50)) / (200 - (-50)), color='red', linewidth=2)  # Adjust the normalization accordingly


    def update(frame):
        print(frame)
        position = states[frame,0:2]
        heading = states[frame,2]        
        theta = np.linspace( 0 , 2 * np.pi , 100 )

        # Clear the previous plot
        ax_asv.clear()
        
        
        if (ship_p.CBF>0) and (ship_p.CBF_plot==1):
            hk = np.ones_like(X)*1000
            for i in range(obs_index):        
                # Initialize hk array
                # Calculate hk for each point in the grid
                for j in range(len(x_range)):
                    for k in range(len(y_range)):
                        x = X[j, k]
                        y = Y[j, k]
                        head_ang = states[frame,2]  # Assume theta = 0 for simplicity
                        v = states[frame,3]  # Assume constant speed for simplicity

                        if (ship_p.CBF==1):
                            B = np.sqrt((x - obs_pos[frame][6*i+0]) ** 2 + (y - obs_pos[frame][6*i+1]) ** 2) - obs_pos[frame][6*i+2]
                            Bdot = ((x - obs_pos[frame][6*i+0]) * v * np.cos(head_ang) + (y - obs_pos[frame][6*i+1]) * v * np.sin(head_ang)) / np.sqrt((x - obs_pos[frame][6*i+0]) ** 2 + (y - obs_pos[frame][6*i+1]) ** 2)
                            hk[j, k] = np.min((hk[j, k], Bdot + ship_p.gamma1/obs_pos[frame][6*i+2] * B))

                        elif (ship_p.CBF==2):
                            R = v/ship_p.rmax*ship_p.gamma_TC1
                            # R = R/2-(R-R/2)*(1-np.cos(np.arctan2((obs_pos[frame][6*i+1]-y),(obs_pos[frame][6*i+0]-x))-head_ang/2))
                            B1 = np.sqrt( (obs_pos[frame][6*i+0]-x-R*np.cos(head_ang-np.pi/2))**2 + (obs_pos[frame][6*i+1]-y-R*np.sin(head_ang-np.pi/2))**2) - (obs_pos[frame][6*i+2]+R)
                            B2 = np.sqrt( (obs_pos[frame][6*i+0]-x-R*np.cos(head_ang+np.pi/2))**2 + (obs_pos[frame][6*i+1]-y-R*np.sin(head_ang+np.pi/2))**2) - (obs_pos[frame][6*i+2]+R)
                            if ship_p.TCCBF == 1:
                                hk[j, k] = np.min((hk[j, k], B1))
                            if ship_p.TCCBF == 2:
                                hk[j, k] = np.min((hk[j, k], B2))
                            if ship_p.TCCBF == 3:
                                hk[j, k] = np.min((hk[j, k], np.max((B1,B2))))
                        if np.abs(np.arctan2((obs_pos[frame][6*i+1]-y),(obs_pos[frame][6*i+0]-x))-head_ang)>np.pi/2:
                            hk[j, k] = 30
            # Plotting
            if ship_p.CBF==2 or ship_p.CBF==1:  # Only add the colorbar once
                ax_asv.contourf(X, Y, hk, levels=np.linspace(hk.min(),hk.max(),15), alpha=0.85, cmap=cm.bone)
                # ax_asv.contourf(X, Y, hk, levels=np.linspace(-50,200,15), alpha=0.85, cmap=cm.bone)                
                # CS = ax_asv.contourf(X, Y, hk, levels=np.linspace(-30,80,10), alpha=0.2, cmap=cm.bone)
                ax_asv.contourf(X, Y, hk, levels=[-0.02, 0.02], colors=['red'], alpha=1)

        # if frame == 0 and ship_p.CBF>=1:  # Only add the colorbar once
        #     # cbar_hk = fig.colorbar(CS, ax=ax_asv, fraction=0.035)
        #     # cbar_hk.set_label("CBF values", fontsize=FS-2)
        #     sm = plt.cm.ScalarMappable(cmap='bone', norm=mcolors.Normalize(vmin=np.min(hk), vmax=np.max(hk)))
        #     sm.set_array([])
        #     cbar_hk = fig.colorbar(sm, ax=ax_asv, fraction=0.035)
        #     cbar_hk.set_label("CBF values", fontsize=FS-2)

        radius = ship_p.radius
        a = position[0] + radius * np.cos( theta )
        b = position[1] + radius * np.sin( theta )    
        ax_asv.fill(a, b, color='green', alpha=0.1)
        
        # Define the vertices of the body connecting the hulls
        body = np.array([[-bodyWidth/2, bodyWidth/2, bodyWidth/2*1.3, bodyWidth/2, -bodyWidth/2, -bodyWidth/2],
                        [separation/2, separation/2, 0, -separation/2, -separation/2, separation/2]])
        # Rotation matrix for the heading
        R = np.array([[np.cos(heading), -np.sin(heading)],
                    [np.sin(heading), np.cos(heading)]])
        # Rotate the body
        body = R @ body
        # Translate the hulls and body to the specified position
        body = body + np.array(position).reshape(2, 1)
        # Calculate the direction vector for the heading
        arrow_length = 0.5
        direction = np.array([np.cos(heading), np.sin(heading)]) * arrow_length        # Plot the ASV
        ax_asv.fill(body[0, :], body[1, :], 'b', alpha=0.3)
        ax_asv.arrow(position[0], position[1], direction[0], direction[1], head_width=0.1, head_length=0.1, fc='k', ec='k')       
        ax_asv.scatter(states[0:frame,0], states[0:frame,1], c=states[0:frame,3], norm=norm, cmap=cm.rainbow, edgecolor='none', marker='o',linewidths=3)        
        # ax_asv.plot(states[0:frame,0], states[0:frame,1], 'k', linewidth=3) 
        ax_asv.plot([states[0,0], states[-1,0]], [0, 0], 'b--')
        ax_asv.plot(position[0], position[1], 'go', linewidth=4)   # Mark the position with a red dot


        a = position[0] + ship_p.radius * np.cos( theta )
        b = position[1] + ship_p.radius * np.sin( theta )    
        ax_asv.fill(a, b, color='green', alpha=(0.2))


        ax_asv.plot(mpc_result[frame][:,0], mpc_result[frame][:,1], 'm--', linewidth=3) 
        ax_asv.xaxis.label.set_size(FS)
        ax_asv.yaxis.label.set_size(FS)


        for jjj in range(0, ship_p.N):
            for i in range(5):
                # i = 4
                radius = obs_pos[frame][6*i+2] - ship_p.radius
                a = obs_pos[frame][6*i+0]+dt*jjj*obs_pos[frame][6*i+3] + radius * np.cos( theta )
                b = obs_pos[frame][6*i+1]+dt*jjj*obs_pos[frame][6*i+4] + radius * np.sin( theta )    
                if jjj == 0:
                    ax_asv.fill(a, b, facecolor='white', hatch='///', edgecolor='black')
                else:
                    # ax_asv.fill(a, b, facecolor='white', hatch='//', edgecolor='black')
                    continue


        ax_asv.set_aspect('equal')
        if mode == 'crossing':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-80,80))
        elif mode == 'avoid':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-30,30))
        elif mode == 'overtaking':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-40,40))
        elif mode == 'static_narrow':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-50,50))
        elif mode == 'static_straight':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-100,100))
        elif mode == 'single_static_straight':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-50,50))
        # ax_asv.grid(True)

        ax_asv.set_xlabel("x [m]", fontsize=FS)
        ax_asv.set_ylabel("y [m]", fontsize=FS)

        times = np.linspace(0, dt*frame, frame)
        # ax_cbf1.clear()
        # ax_cbf2.clear()
        ax_cbf3.clear()
        ax_compt.clear()

        ax_compt.grid(True)
        for j in range(obs_index):
            ax_compt.plot(times,cbf_and_dist[0:frame,j])
        ax_compt.plot([0, len(cbf_and_dist)*dt],[0,0],'r--')
        ax_compt.set_xlim([0, len(cbf_and_dist)*dt])
        ax_compt.set_ylim([-1, 10])
        ax_compt.set_xlabel("Time", fontsize=FS)
        ax_compt.set_ylabel("CBF", fontsize=FS)
        
        # ax_cbf2.grid(True)
        # for j in range(5):
        #     if ship_p.CBF == 1:
        #         ax_cbf2.plot(times,cbf_and_dist[1:frame+1,j]-(1-ship_p.gamma2)*cbf_and_dist[0:frame,j])
        #     if ship_p.CBF == 2:
        #         ax_cbf2.plot(times,cbf_and_dist[1:frame+1,j]-(1-ship_p.gamma_TC2)*cbf_and_dist[0:frame,j])
        # ax_cbf2.plot([0, len(cbf_and_dist)*dt],[0,0],'r--')
        # ax_cbf2.set_xlim([0, len(cbf_and_dist)*dt])
        # ax_cbf2.set_ylim([-0.5, 3])
        # ax_cbf2.set_xlabel("Time", fontsize=FS)
        # ax_cbf2.set_ylabel("CBF dot", fontsize=FS)
        
        ax_cbf3.grid(True)
        for j in range(obs_index):            
            ax_cbf3.plot(times,cbf_and_dist[0:frame,5+j])
        ax_cbf3.plot([0, len(cbf_and_dist)*dt],[0,0],'r--')
        ax_cbf3.set_xlim([0, len(cbf_and_dist)*dt])
        # ax_cbf3.set_ylim([-5, np.max(cbf_and_dist[:,0])])
        ax_cbf3.set_ylim([-1, 10])
        ax_cbf3.set_xlabel("Time", fontsize=FS)
        ax_cbf3.set_ylabel("Closest Distance", fontsize=FS)

        # ax_compt.grid(True)
        # ax_compt.plot(times,comptime[0:frame])
        # ax_compt.set_xlim([0, len(comptime)*dt])
        # ax_compt.set_ylim([-0, np.max(comptime[:])])
        # ax_compt.set_xlabel("Time", fontsize=FS)
        # ax_compt.set_ylabel("Comp. time", fontsize=FS)

        # ax_cbf1.xaxis.label.set_size(FS)
        # ax_cbf1.yaxis.label.set_size(FS)

        # ax_cbf2.xaxis.label.set_size(FS)static_straight
        # ax_cbf2.yaxis.label.set_size(FS)

        ax_cbf3.xaxis.label.set_size(FS)
        ax_cbf3.yaxis.label.set_size(FS)

        ax_compt.xaxis.label.set_size(FS)
        ax_compt.yaxis.label.set_size(FS)

        t = np.arange(0, dt*frame, dt)
        ax_speed.plot(t, states[0:frame,3], 'k', linewidth=2) 
        ax_speed.plot(t, states[0:frame,3]*0 + target_speed, 'b--', linewidth=2) 
        ax_speed.set_xlabel("Time", fontsize=FS)
        ax_speed.set_ylabel("Speed [m/s]", fontsize=FS)
        ax_speed.grid(True)
        ax_speed.autoscale(enable=True, axis='x', tight=True)
        ax_speed.autoscale(enable=True, axis='y', tight=True)
        ax_speed.set_ylim(0.5, 2.5)

        # Plot the headings
        ax_rot.plot(t, states[0:frame,4], 'k', linewidth=2) 
        ax_rot.set_xlabel("Time", fontsize=FS)
        ax_rot.set_ylabel("Rot. Speed [rad/s]", fontsize=FS)
        ax_rot.grid(True)
        ax_rot.autoscale(enable=True, axis='x', tight=True)
        ax_rot.autoscale(enable=True, axis='y', tight=True)
        ax_rot.set_ylim(-0.2, 0.2)

        # Plot the figure-eight trajectory
        ax_thrust.plot(t, states[0:frame,5], 'r')
        ax_thrust.plot(t, states[0:frame,6], 'g')
        ax_thrust.plot(t, states[0:frame,5]*0 + ship_p.Fxmax, 'r--')
        ax_thrust.plot(t, states[0:frame,5]*0 + ship_p.Fxmin, 'r--')
        ax_thrust.set_xlabel("Time", fontsize=FS)
        ax_thrust.set_ylabel("Thrust", fontsize=FS)
        ax_thrust.grid(True)
        ax_thrust.autoscale(enable=True, axis='x', tight=True)
        ax_thrust.set_ylim(ship_p.Fxmin-1, ship_p.Fxmax+1)

        fig.tight_layout()  # axes 사이 간격을 적당히 벌려줍니다.

    frames = range(0, len(states), plot_iter)
    if plot_iter == len(states)-1:
        frames = range(len(states)-1, len(states))
    anim = FuncAnimation(fig, update, frames, init_function, repeat=False)
    # plt.show()
    if ship_p.CBF == 2:
        anim.save('Result_' + mode + '_cbf_type_' + str(ship_p.CBF) + '_TCCBF_type_=' + str(ship_p.TCCBF) + '_N=' + str(ship_p.N) + '.mp4', writer=animation.FFMpegWriter(fps=20))  
    elif ship_p.CBF == 1:
        anim.save('Result_' + mode + '_cbf_type_' + str(ship_p.CBF) + '_EDCBF_gamma1_=' + str(ship_p.gamma1) + '_N=' + str(ship_p.N) + '.mp4', writer=animation.FFMpegWriter(fps=20))          
    else:
        anim.save('Result_' + mode + 'cbf_type_' + str(ship_p.CBF) + 'N=' + str(ship_p.N) + '.mp4', writer=animation.FFMpegWriter(fps=20))  


def init_function():
    return



##########################################

def plot_inputs(t, states, inputs, target_speed, mode):
    FS = 18
    ship_p = load_ship_param
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    # Plot the figure-eight trajectory
    axs[0, 0].plot(t, states[0:-1,3], 'k', linewidth=2) 
    axs[0, 0].plot(t, states[0:-1,3]*0 + target_speed, 'b--', linewidth=2) 
    axs[0, 0].set_xlabel("Time", fontsize=FS)
    axs[0, 0].set_ylabel("Speed [m/s]", fontsize=FS)
    axs[0, 0].grid(True)
    axs[0, 0].autoscale(enable=True, axis='x', tight=True)
    axs[0, 0].autoscale(enable=True, axis='y', tight=True)
    axs[0, 0].set_ylim(1.2, 2.1)

    # Plot the headings
    axs[0, 1].plot(t, states[0:-1,4], 'k', linewidth=2) 
    axs[0, 1].set_xlabel("Time", fontsize=FS)
    axs[0, 1].set_ylabel("Rot. Speed [rad/s]", fontsize=FS)
    axs[0, 1].grid(True)
    axs[0, 1].autoscale(enable=True, axis='x', tight=True)
    axs[0, 1].autoscale(enable=True, axis='y', tight=True)
    axs[0, 1].set_ylim(-0.2, 0.2)

    # Plot the figure-eight trajectory
    axs[1, 0].plot(t, states[0:-1,5], 'r')
    axs[1, 0].plot(t, states[0:-1,6], 'g')
    axs[1, 0].plot(t, states[0:-1,5]*0 + ship_p.Fxmax, 'r--')
    axs[1, 0].plot(t, states[0:-1,5]*0 + ship_p.Fxmin, 'r--')
    axs[1, 0].set_xlabel("Time", fontsize=FS)
    axs[1, 0].set_ylabel("Thrust", fontsize=FS)
    axs[1, 0].grid(True)
    axs[1, 0].autoscale(enable=True, axis='x', tight=True)
    axs[1, 0].set_ylim(ship_p.Fxmin-5, ship_p.Fxmax+5   )

    # Plot the headings
    # axs[1, 1].plot(t, states[0:-1,5]+states[0:-1,6], 'c')
    # axs[1, 1].plot(t, (-states[0:-1,5]+states[0:-1,6])*ship_p.L/2, 'm')
    axs[1, 1].plot(t, inputs[0:-1,0], 'r')
    axs[1, 1].plot(t, inputs[0:-1,1], 'g')
    axs[1, 1].plot(t, states[0:-1,6]*0 + ship_p.dFnmax, 'r--')
    axs[1, 1].plot(t, states[0:-1,6]*0 - ship_p.dFnmax, 'r--')
    axs[1, 1].set_xlabel("Time", fontsize=FS)
    axs[1, 1].set_ylabel("del. Thrust", fontsize=FS)
    axs[1, 1].grid(True)
    axs[1, 1].autoscale(enable=True, axis='x', tight=True)
    axs[1, 1].set_ylim(-ship_p.dFnmax-1, ship_p.dFnmax+1)

    # # Plot the figure-eight trajectory
    # axs[0, 2].plot(t, inputs[0:-1,0], 'k')
    # axs[0, 2].plot(t, inputs[0:-1,0]*0 + ship_p.dFxmax, 'r--')
    # axs[0, 2].plot(t, inputs[0:-1,0]*0 - ship_p.dFxmax, 'r--')
    # axs[0, 2].set_xlabel("Time", fontsize=FS)
    # axs[0, 2].set_ylabel("del. Fx-Thrust", fontsize=FS)
    # axs[0, 2].grid(True)
    # axs[0, 2].axis('equal')
    # axs[0, 2].autoscale(enable=True, axis='x', tight=True)
    # axs[0, 2].set_ylim(-ship_p.dFxmax-1, ship_p.dFxmax+1)
    

    # Plot the headings
    # axs[1, 2].plot(t, inputs[0:-1,1], 'k')
    # axs[1, 2].plot(t, inputs[0:-1,1]*0 + ship_p.dFnmax, 'r--')
    # axs[1, 2].plot(t, inputs[0:-1,1]*0 - ship_p.dFnmax, 'r--')
    # axs[1, 2].set_xlabel("Time", fontsize=FS)
    # axs[1, 2].set_ylabel("del. Fn-Moment", fontsize=FS)
    # axs[1, 2].grid(True)
    # axs[1, 2].axis('equal')
    # axs[1, 2].autoscale(enable=True, axis='x', tight=True)
    # axs[1, 2].set_ylim(-ship_p.dFnmax-1, ship_p.dFnmax+1)

    plt.tight_layout()
    date_string = time.strftime("%Y-%m-%d-%H:%M")
    # plt.savefig(date_string + 'cbf_type_' + str(ship_p.CBF) + '.png')
    plt.savefig('input_' + mode + 'cbf_type_' + str(ship_p.CBF) + 'N=' + str(ship_p.N) + '.png')
    # plt.savefig('input_' + 'avoid' + 'cbf_type_' + str(ship_p.CBF) + 'N=' + str(ship_p.N) + '.png')
    plt.close(fig)  # Close the figure to free memory
    # plt.show()




# def animateCBF(plot_iter, cbf_and_dist, mode):
    
#     fig, ax = plt.subplots(1,3,figsize=(15,6))
#     ship_p = load_ship_param
#     dt = ship_p.dt
#     def update(frame):

#         times = np.linspace(0, dt*frame, frame)

#         ax[0].clear()
#         ax[1].clear()
#         ax[2].clear()

#         ax[0].grid(True)
#         for j in range(5):
#             ax[0].plot(times,cbf_and_dist[0:frame,j])
#         ax[0].plot([0, len(cbf_and_dist)*dt],[0,0],'r--')
#         ax[0].set_xlim([0, len(cbf_and_dist)*dt])
#         ax[0].set_ylim([-25, np.max(cbf_and_dist[:,0])])
#         ax[0].set_xlabel("Time", fontsize=14)
#         ax[0].set_ylabel("CBF", fontsize=14)
        
#         ax[1].grid(True)
#         for j in range(5):
#             if ship_p.CBF == 1:
#                 ax[1].plot(times,cbf_and_dist[1:frame+1,j]-(1-ship_p.gamma2)*cbf_and_dist[0:frame,j])
#             if ship_p.CBF == 2:
#                 ax[1].plot(times,cbf_and_dist[1:frame+1,j]-(1-ship_p.gamma_TC2)*cbf_and_dist[0:frame,j])
#         ax[1].plot([0, len(cbf_and_dist)*dt],[0,0],'r--')
#         ax[1].set_xlim([0, len(cbf_and_dist)*dt])
#         ax[1].set_ylim([-1, 3])
#         ax[1].set_xlabel("Time", fontsize=14)
#         ax[1].set_ylabel("CBF", fontsize=14)
        
#         ax[2].grid(True)
#         for j in range(5):            
#             ax[2].plot(times,cbf_and_dist[0:frame,5+j])
#         ax[2].plot([0, len(cbf_and_dist)*dt],[0,0],'r--')
#         ax[2].set_xlim([0, len(cbf_and_dist)*dt])
#         ax[2].set_ylim([-15, np.max(cbf_and_dist[:,0])])
#         ax[2].set_xlabel("Time", fontsize=14)
#         ax[2].set_ylabel("Closest Distance", fontsize=14)
            
#     frames = range(0, len(cbf_and_dist), plot_iter)
#     anim = FuncAnimation(fig, update, frames, repeat=False)
#     date_string = time.strftime("%Y-%m-%d-%H:%M")
#     # anim.save(date_string + 'cbf_type_' + str(ship_p.CBF) + '.gif', writer='imagemagick')  # Save the animation as a gif file
#     # anim.save('CBF' + mode + 'cbf_type_' + str(ship_p.CBF) + 'N=' + str(ship_p.N) + '.gif', writer=PillowWriter(fps=20))  
#     anim.save('CBF' + mode + 'cbf_type_' + str(2) + 'N=' + str(5) + '.gif', writer=PillowWriter(fps=20))  
#     plt.close(fig)  # Close the figure to free memory    
