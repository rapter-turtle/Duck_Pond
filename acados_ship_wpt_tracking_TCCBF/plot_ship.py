import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from load_param import load_ship_param
from matplotlib import cm
from matplotlib.colors import Normalize
import time

def animateASV(states, target_speed, mpc_result, obs_pos, cbf_and_dist, plot_iter, comptime, mode):
    
    fig, axs = plt.subplots(2,3, figsize=(18,11))
    
    # Combine first two columns for the ASV plot
    fig.delaxes(axs[0, 2])
    ax_asv = plt.subplot2grid((2, 3), (0, 1), colspan=2)  # Combine first row for ASV
    # CBF animation setup
    ax_compt = axs[0, 0]
    ax_cbf1 = axs[1, 0]
    ax_cbf2 = axs[1, 1]
    ax_cbf3 = axs[1, 2]


    ship_p = load_ship_param
    dt = ship_p.dt

    # Define the geometry of the twin-hull ASV
    separation = ship_p.L*0.3  # Distance between the two hulls
    bodyWidth = ship_p.L  # Width of the body connecting the hulls

    # Define the grid for plotting
    x_range = np.linspace(states[0,0], states[-1,0], 200)
    y_range = np.linspace(-80, 80, 200)
    X, Y = np.meshgrid(x_range, y_range)
    

    # Create a scatter plot for the heatmap and initialize the colorbar
    norm = Normalize(vmin=np.min(states[:, 3]), vmax=np.max(states[:, 3]))
    norm = Normalize(vmin=1, vmax=3)
    heatmap = ax_asv.scatter(states[:, 0], states[:, 1], c=states[:, 3], norm=norm, cmap=cm.rainbow, edgecolor='none', marker='o')
    cbar = plt.colorbar(heatmap, ax=ax_asv, fraction=0.035)
    cbar.set_label("velocity in [m/s]")

    FS = 14

    def update(frame):
        print(frame)
        position = states[frame,0:2]
        heading = states[frame,2]        
        theta = np.linspace( 0 , 2 * np.pi , 100 )

        # Clear the previous plot
        ax_asv.clear()
        
        
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
        ax_asv.scatter(states[0:frame,0], states[0:frame,1], c=states[0:frame,3], norm=norm, cmap=cm.rainbow, edgecolor='none', marker='o')        
        ax_asv.plot([states[0,0]-20, states[-1,0]+20], [0, 0], 'k--')
        ax_asv.plot(position[0], position[1], 'go')  # Mark the position with a red dot
        ax_asv.axis('equal')
        ax_asv.plot(mpc_result[frame][:,0], mpc_result[frame][:,1], 'k.')
        ax_asv.xaxis.label.set_size(FS)
        ax_asv.yaxis.label.set_size(FS)


        
        for i in range(5):
            radius = obs_pos[frame][5*i+2] - ship_p.radius
            a = obs_pos[frame][5*i+0] + radius * np.cos( theta )
            b = obs_pos[frame][5*i+1] + radius * np.sin( theta )    
            ax_asv.fill(a, b, color='red', alpha=0.3)

            radius = obs_pos[frame][5*i+2]
            a = obs_pos[frame][5*i+0] + radius * np.cos( theta )
            b = obs_pos[frame][5*i+1] + radius * np.sin( theta )    
            ax_asv.fill(a, b, color='red', alpha=0.1)

        if (ship_p.CBF>0) and (ship_p.CBF_plot==1):
            for i in range(5):        
                # Initialize hk array
                hk = np.zeros_like(X)
                # Calculate hk for each point in the grid
                for j in range(len(x_range)):
                    for k in range(len(y_range)):
                        x = X[j, k]
                        y = Y[j, k]
                        theta = states[frame,2]  # Assume theta = 0 for simplicity
                        v = states[frame,3]  # Assume constant speed for simplicity

                        if (ship_p.CBF==1):
                            B = np.sqrt((x - obs_pos[frame][5*i+0]) ** 2 + (y - obs_pos[frame][5*i+1]) ** 2) - obs_pos[frame][5*i+2]
                            Bdot = ((x - obs_pos[frame][5*i+0]) * v * np.cos(theta) + (y - obs_pos[frame][5*i+1]) * v * np.sin(theta)) / np.sqrt((x - obs_pos[frame][5*i+0]) ** 2 + (y - obs_pos[frame][5*i+1]) ** 2)
                            hk[j, k] = Bdot + ship_p.gamma1/obs_pos[frame][5*i+2] * B

                        elif (ship_p.CBF==2):
                            R = v/ship_p.rmax*ship_p.gamma_TC1
                            B1 = np.sqrt( (obs_pos[frame][5*i+0]-x-R*np.cos(theta-np.pi/2))**2 + (obs_pos[frame][5*i+1]-y-R*np.sin(theta-np.pi/2))**2) - (obs_pos[frame][5*i+2]+R)
                            B2 = np.sqrt( (obs_pos[frame][5*i+0]-x-R*np.cos(theta+np.pi/2))**2 + (obs_pos[frame][5*i+1]-y-R*np.sin(theta+np.pi/2))**2) - (obs_pos[frame][5*i+2]+R)
                            hk[j, k] = np.max((B1,B2))
                            if np.abs(np.arctan2((obs_pos[frame][5*i+1]-y),(obs_pos[frame][5*i+0]-x))-theta)>np.pi/2:
                                hk[j, k] = 100
                            # hk[j, k] = B1                            

                # Plotting
                ax_asv.contourf(X, Y, hk, levels=[-np.inf, 0], colors=['orange'], alpha=0.3)

        # ax.axis([-15, 15, -15, 15])  # Adjust these limits as needed
        # ax.axis([-15, 145, -15, 15])  # Adjust these limits as needed

        ax_asv.autoscale(enable=True, axis='x', tight=True)
        ax_asv.autoscale(enable=True, axis='y', tight=True)
        
        # Axis 범위 지정
        ax_asv.set_xlim([states[0, 0] - 50, states[-1, 0] + 50])
        ax_asv.set_ylim([-50, 50])
        ax_asv.grid(True)


        times = np.linspace(0, dt*frame, frame)
        ax_cbf1.clear()
        ax_cbf2.clear()
        ax_cbf3.clear()
        ax_compt.clear()

        ax_cbf1.grid(True)
        for j in range(5):
            ax_cbf1.plot(times,cbf_and_dist[0:frame,j])
        ax_cbf1.plot([0, len(cbf_and_dist)*dt],[0,0],'r--')
        ax_cbf1.set_xlim([0, len(cbf_and_dist)*dt])
        ax_cbf1.set_ylim([-10, 30])
        ax_cbf1.set_xlabel("Time", fontsize=FS)
        ax_cbf1.set_title("CBF", fontsize=FS)
        
        ax_cbf2.grid(True)
        for j in range(5):
            if ship_p.CBF == 1:
                ax_cbf2.plot(times,cbf_and_dist[1:frame+1,j]-(1-ship_p.gamma2)*cbf_and_dist[0:frame,j])
            if ship_p.CBF == 2:
                ax_cbf2.plot(times,cbf_and_dist[1:frame+1,j]-(1-ship_p.gamma_TC2)*cbf_and_dist[0:frame,j])
        ax_cbf2.plot([0, len(cbf_and_dist)*dt],[0,0],'r--')
        ax_cbf2.set_xlim([0, len(cbf_and_dist)*dt])
        ax_cbf2.set_ylim([-2, 3])
        ax_cbf2.set_xlabel("Time", fontsize=FS)
        ax_cbf2.set_title("DCBF", fontsize=FS)
        
        ax_cbf3.grid(True)
        for j in range(5):            
            ax_cbf3.plot(times,cbf_and_dist[0:frame,5+j])
        ax_cbf3.plot([0, len(cbf_and_dist)*dt],[0,0],'r--')
        ax_cbf3.set_xlim([0, len(cbf_and_dist)*dt])
        # ax_cbf3.set_ylim([-5, np.max(cbf_and_dist[:,0])])
        ax_cbf3.set_ylim([-5, 20])
        ax_cbf3.set_xlabel("Time", fontsize=FS)
        ax_cbf3.set_title("Closest Distance", fontsize=FS)

        ax_compt.grid(True)
        ax_compt.plot(times,comptime[0:frame])
        ax_compt.set_xlim([0, len(comptime)*dt])
        ax_compt.set_ylim([-0, np.max(comptime[:])])
        ax_compt.set_xlabel("Time", fontsize=FS)
        ax_compt.set_title("Comp. time", fontsize=FS)

        ax_cbf1.xaxis.label.set_size(FS)
        ax_cbf1.yaxis.label.set_size(FS)

        ax_cbf2.xaxis.label.set_size(FS)
        ax_cbf2.yaxis.label.set_size(FS)

        ax_cbf3.xaxis.label.set_size(FS)
        ax_cbf3.yaxis.label.set_size(FS)

        ax_compt.xaxis.label.set_size(FS)
        ax_compt.yaxis.label.set_size(FS)

        

    frames = range(0, len(states), plot_iter)
    anim = FuncAnimation(fig, update, frames, repeat=False)
    date_string = time.strftime("%Y-%m-%d-%H:%M")
    # anim.save(date_string + 'cbf_type_' + str(ship_p.CBF) + '.gif', writer='imagemagick')  # Save the animation as a gif file
    anim.save('Result_' + mode + 'cbf_type_' + str(ship_p.CBF) + 'N=' + str(ship_p.N) + '.gif', writer=PillowWriter(fps=20))  
    plt.close(fig)  # Close the figure to free memory



def animateCBF(plot_iter, cbf_and_dist, mode):
    
    fig, ax = plt.subplots(1,3,figsize=(15,6))
    ship_p = load_ship_param
    dt = ship_p.dt
    def update(frame):

        times = np.linspace(0, dt*frame, frame)

        ax[0].clear()
        ax[1].clear()
        ax[2].clear()

        ax[0].grid(True)
        for j in range(5):
            ax[0].plot(times,cbf_and_dist[0:frame,j])
        ax[0].plot([0, len(cbf_and_dist)*dt],[0,0],'r--')
        ax[0].set_xlim([0, len(cbf_and_dist)*dt])
        ax[0].set_ylim([-25, np.max(cbf_and_dist[:,0])])
        ax[0].set_xlabel("Time", fontsize=14)
        ax[0].set_ylabel("CBF", fontsize=14)
        
        ax[1].grid(True)
        for j in range(5):
            if ship_p.CBF == 1:
                ax[1].plot(times,cbf_and_dist[1:frame+1,j]-(1-ship_p.gamma2)*cbf_and_dist[0:frame,j])
            if ship_p.CBF == 2:
                ax[1].plot(times,cbf_and_dist[1:frame+1,j]-(1-ship_p.gamma_TC2)*cbf_and_dist[0:frame,j])
        ax[1].plot([0, len(cbf_and_dist)*dt],[0,0],'r--')
        ax[1].set_xlim([0, len(cbf_and_dist)*dt])
        ax[1].set_ylim([-1, 3])
        ax[1].set_xlabel("Time", fontsize=14)
        ax[1].set_ylabel("CBF", fontsize=14)
        
        ax[2].grid(True)
        for j in range(5):            
            ax[2].plot(times,cbf_and_dist[0:frame,5+j])
        ax[2].plot([0, len(cbf_and_dist)*dt],[0,0],'r--')
        ax[2].set_xlim([0, len(cbf_and_dist)*dt])
        ax[2].set_ylim([-15, np.max(cbf_and_dist[:,0])])
        ax[2].set_xlabel("Time", fontsize=14)
        ax[2].set_ylabel("Closest Distance", fontsize=14)
            
    frames = range(0, len(cbf_and_dist), plot_iter)
    anim = FuncAnimation(fig, update, frames, repeat=False)
    date_string = time.strftime("%Y-%m-%d-%H:%M")
    # anim.save(date_string + 'cbf_type_' + str(ship_p.CBF) + '.gif', writer='imagemagick')  # Save the animation as a gif file
    # anim.save('CBF' + mode + 'cbf_type_' + str(ship_p.CBF) + 'N=' + str(ship_p.N) + '.gif', writer=PillowWriter(fps=20))  
    anim.save('CBF' + mode + 'cbf_type_' + str(ship_p.CBF) + 'N=' + str(5) + '.gif', writer=PillowWriter(fps=20))  
    plt.close(fig)  # Close the figure to free memory    

def plot_inputs(t, states, inputs, target_speed, mode):
    FS = 18
    ship_p = load_ship_param
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    # Plot the figure-eight trajectory
    axs[0, 0].plot(t, states[0:-1,3], 'k', linewidth=2) 
    axs[0, 0].plot(t, states[0:-1,3]*0 + target_speed, 'b--', linewidth=2) 
    axs[0, 0].set_xlabel("Time", fontsize=FS)
    axs[0, 0].set_ylabel("Speed [m/s]", fontsize=FS)
    axs[0, 0].grid(True)
    axs[0, 0].autoscale(enable=True, axis='x', tight=True)
    axs[0, 0].autoscale(enable=True, axis='y', tight=True)

    # Plot the headings
    axs[0, 1].plot(t, states[0:-1,4], 'k', linewidth=2) 
    axs[0, 1].set_xlabel("Time", fontsize=FS)
    axs[0, 1].set_ylabel("Rot. Speed [rad/s]", fontsize=FS)
    axs[0, 1].grid(True)
    axs[0, 1].autoscale(enable=True, axis='x', tight=True)
    axs[0, 1].autoscale(enable=True, axis='y', tight=True)

    # Plot the figure-eight trajectory
    axs[1, 0].plot(t, states[0:-1,5], 'k')
    axs[1, 0].plot(t, states[0:-1,5]*0 + ship_p.Fxmax, 'r--')
    axs[1, 0].plot(t, states[0:-1,5]*0 + ship_p.Fxmin, 'r--')
    axs[1, 0].set_xlabel("Time", fontsize=FS)
    axs[1, 0].set_ylabel("Fx-Thrust", fontsize=FS)
    axs[1, 0].grid(True)
    axs[1, 0].autoscale(enable=True, axis='x', tight=True)
    axs[1, 0].set_ylim(ship_p.Fxmin-10, ship_p.Fxmax+10)

    # Plot the headings
    axs[1, 1].plot(t, states[0:-1,6], 'k')
    axs[1, 1].plot(t, states[0:-1,6]*0 + ship_p.Fnmax, 'r--')
    axs[1, 1].plot(t, states[0:-1,6]*0 - ship_p.Fnmax, 'r--')
    axs[1, 1].set_xlabel("Time", fontsize=FS)
    axs[1, 1].set_ylabel("Fn-Moment", fontsize=FS)
    axs[1, 1].grid(True)
    axs[1, 1].autoscale(enable=True, axis='x', tight=True)
    axs[1, 1].set_ylim(-ship_p.Fnmax-10, ship_p.Fnmax+10)

    # Plot the figure-eight trajectory
    axs[0, 2].plot(t, inputs[0:-1,0], 'k')
    axs[0, 2].plot(t, inputs[0:-1,0]*0 + ship_p.dFxmax, 'r--')
    axs[0, 2].plot(t, inputs[0:-1,0]*0 - ship_p.dFxmax, 'r--')
    axs[0, 2].set_xlabel("Time", fontsize=FS)
    axs[0, 2].set_ylabel("del. Fx-Thrust", fontsize=FS)
    axs[0, 2].grid(True)
    axs[0, 2].autoscale(enable=True, axis='x', tight=True)
    axs[0, 2].set_ylim(-ship_p.dFxmax-1, ship_p.dFxmax+1)

    # Plot the headings
    axs[1, 2].plot(t, inputs[0:-1,1], 'k')
    axs[1, 2].plot(t, inputs[0:-1,1]*0 + ship_p.dFnmax, 'r--')
    axs[1, 2].plot(t, inputs[0:-1,1]*0 - ship_p.dFnmax, 'r--')
    axs[1, 2].set_xlabel("Time", fontsize=FS)
    axs[1, 2].set_ylabel("del. Fn-Moment", fontsize=FS)
    axs[1, 2].grid(True)
    axs[1, 2].autoscale(enable=True, axis='x', tight=True)
    axs[1, 2].set_ylim(-ship_p.dFnmax-1, ship_p.dFnmax+1)

    plt.tight_layout()
    date_string = time.strftime("%Y-%m-%d-%H:%M")
    # plt.savefig(date_string + 'cbf_type_' + str(ship_p.CBF) + '.png')
    plt.savefig('input_' + mode + 'cbf_type_' + str(ship_p.CBF) + 'N=' + str(ship_p.N) + '.png')
    plt.close(fig)  # Close the figure to free memory
    # plt.show()
