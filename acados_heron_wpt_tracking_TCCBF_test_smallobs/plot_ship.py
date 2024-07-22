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
    y_range = np.linspace(-20, 20, 200)
    

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
                            B = np.sqrt((x - obs_pos[frame][5*i+0]) ** 2 + (y - obs_pos[frame][5*i+1]) ** 2) - obs_pos[frame][5*i+2]
                            Bdot = ((x - obs_pos[frame][5*i+0]) * v * np.cos(head_ang) + (y - obs_pos[frame][5*i+1]) * v * np.sin(head_ang)) / np.sqrt((x - obs_pos[frame][5*i+0]) ** 2 + (y - obs_pos[frame][5*i+1]) ** 2)
                            hk[j, k] = np.min((hk[j, k], Bdot + ship_p.gamma1/obs_pos[frame][5*i+2] * B))

                        elif (ship_p.CBF==2):
                            R = v/ship_p.rmax*ship_p.gamma_TC1
                            # R = R/2-(R-R/2)*(1-np.cos(np.arctan2((obs_pos[frame][5*i+1]-y),(obs_pos[frame][5*i+0]-x))-head_ang/2))
                            B1 = np.sqrt( (obs_pos[frame][5*i+0]-x-R*np.cos(head_ang-np.pi/2))**2 + (obs_pos[frame][5*i+1]-y-R*np.sin(head_ang-np.pi/2))**2) - (obs_pos[frame][5*i+2]+R)
                            B2 = np.sqrt( (obs_pos[frame][5*i+0]-x-R*np.cos(head_ang+np.pi/2))**2 + (obs_pos[frame][5*i+1]-y-R*np.sin(head_ang+np.pi/2))**2) - (obs_pos[frame][5*i+2]+R)
                            if ship_p.TCCBF == 1:
                                hk[j, k] = np.min((hk[j, k], B1))
                            if ship_p.TCCBF == 2:
                                hk[j, k] = np.min((hk[j, k], B2))
                            if ship_p.TCCBF == 3:
                                hk[j, k] = np.min((hk[j, k], np.max((B1,B2))))
                        if np.abs(np.arctan2((obs_pos[frame][5*i+1]-y),(obs_pos[frame][5*i+0]-x))-head_ang)>np.pi/2:
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
                radius = obs_pos[frame][5*i+2] - ship_p.radius
                a = obs_pos[frame][5*i+0]+dt*jjj*obs_pos[frame][5*i+3] + radius * np.cos( theta )
                b = obs_pos[frame][5*i+1]+dt*jjj*obs_pos[frame][5*i+4] + radius * np.sin( theta )    
                if jjj == 0:
                    ax_asv.fill(a, b, facecolor='white', hatch='///', edgecolor='black')
                else:
                    # ax_asv.fill(a, b, facecolor='white', hatch='//', edgecolor='black')
                    continue


        ax_asv.set_aspect('equal')
        if mode == 'avoid':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-5,5))
        elif mode == 'overtaking':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-7.5,7.5))
        elif mode == 'static_narrow':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-6,6))
        elif mode == 'static_straight':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-10,10))
        elif mode == 'single_static_straight':
            ax_asv.set(xlim=(states[0, 0], states[-1, 0]),ylim=(-10,10))
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
        ax_rot.plot(t, states[0:frame,4]*0 - ship_p.rmax, 'r--')
        ax_rot.plot(t, states[0:frame,4]*0 + ship_p.rmax, 'r--')
        ax_rot.set_xlabel("Time", fontsize=FS)
        ax_rot.set_ylabel("Rot. Speed [rad/s]", fontsize=FS)
        ax_rot.grid(True)
        ax_rot.autoscale(enable=True, axis='x', tight=True)
        ax_rot.autoscale(enable=True, axis='y', tight=True)
        ax_rot.set_ylim(-ship_p.rmax-0.01, ship_p.rmax+0.01)

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

