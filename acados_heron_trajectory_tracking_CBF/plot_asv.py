import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from load_param import load_param
from matplotlib import cm
from matplotlib.colors import Normalize

def animateASV(states, inputs, ref, yref, mpc_result, obs_pos, plot_iter):
    # Define the geometry of the twin-hull ASV
    hullLength = 0.7  # Length of the hull
    hullWidth = 0.2   # Width of each hull
    separation = 0.4  # Distance between the two hulls
    bodyLength = hullLength  # Length of the body connecting the hulls
    bodyWidth = 0.25  # Width of the body connecting the hulls

    fig, ax = plt.subplots()

    heron_p = load_param


    # Define the grid for plotting
    x_range = np.linspace(-20, 20, 200)
    y_range = np.linspace(-20, 20, 200)
    X, Y = np.meshgrid(x_range, y_range)
    
    
    # Create a scatter plot for the heatmap and initialize the colorbar
    norm = Normalize(vmin=np.min(states[:, 3]), vmax=np.max(states[:, 3]))
    heatmap = ax.scatter(states[:, 0], states[:, 1], c=states[:, 3], norm=norm, cmap=cm.rainbow, edgecolor='none', marker='o')
    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.035)
    cbar.set_label("velocity in [m/s]")

    def update(frame):
        position = states[frame,0:2]
        heading = states[frame,2]
        vels = states[frame,3]
        print(inputs[frame,0:2])
        print()
        force_left = states[frame,5]/100
        force_right = states[frame,6]/100
        # Clear the previous plot
        ax.clear()
        
        # ax.plot(states[0:frame,0], states[0:frame,1], 'b-')  # Mark the position with a red dot
        ax.scatter(states[0:frame,0], states[0:frame,1], c=states[0:frame,3], norm=norm, cmap=cm.rainbow, edgecolor='none', marker='o')

        # Define the vertices of the two hulls
        hull1 = np.array([[-hullLength/2, hullLength/2, hullLength/2, -hullLength/2, -hullLength/2, -hullLength/2],
                        [hullWidth/2, hullWidth/2, -hullWidth/2, -hullWidth/2, 0, hullWidth/2]])

        hull2 = np.array([[-hullLength/2, hullLength/2, hullLength/2, -hullLength/2, -hullLength/2, -hullLength/2],
                        [hullWidth/2, hullWidth/2, -hullWidth/2, -hullWidth/2, 0, hullWidth/2]])

        # Define the vertices of the body connecting the hulls
        body = np.array([[-bodyWidth/2, bodyWidth/2, bodyWidth/2, -bodyWidth/2, -bodyWidth/2],
                        [separation/2, separation/2, -separation/2, -separation/2, separation/2]])

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


        # Calculate the positions for the force vectors
        force_left_position = hull1[:, 4]
        force_right_position = hull2[:, 4]

        # Plot the force vectors
        if force_left > 0:
            ax.arrow(force_left_position[0], force_left_position[1], force_left*np.cos(heading), force_left*np.sin(heading), head_width=0.1, head_length=0.1, fc='b', ec='b')
        else:
            ax.arrow(force_left_position[0], force_left_position[1], force_left*np.cos(heading), force_left*np.sin(heading), head_width=0.1, head_length=0.1, fc='r', ec='r')
        
        if force_right > 0:
            ax.arrow(force_right_position[0], force_right_position[1], force_right*np.cos(heading), force_right*np.sin(heading), head_width=0.1, head_length=0.1, fc='b', ec='b')
        else:
            ax.arrow(force_right_position[0], force_right_position[1], force_right*np.cos(heading), force_right*np.sin(heading), head_width=0.1, head_length=0.1, fc='r', ec='r')

        # Plot the ASV
        ax.fill(hull1[0, :], hull1[1, :], 'b', alpha=0.3)
        ax.fill(hull2[0, :], hull2[1, :], 'b', alpha=0.3)
        ax.fill(body[0, :], body[1, :], 'b', alpha=0.3)
        
        ax.plot(position[0], position[1], 'go')  # Mark the position with a red dot
        ax.arrow(position[0], position[1], direction[0], direction[1], head_width=0.1, head_length=0.1, fc='k', ec='k')
        ax.axis('equal')

        ax.plot(ref[:,0], ref[:,1], 'k--')
        ax.plot(yref[frame,:,0], yref[frame,:,1], 'c.')
        ax.plot(mpc_result[frame][:,0], mpc_result[frame][:,1], 'k.')


        theta = np.linspace( 0 , 2 * np.pi , 150 )
        
        for i in range(5):
            radius = obs_pos[frame][3*i+2]        
            a = obs_pos[frame][3*i+0] + radius * np.cos( theta )
            b = obs_pos[frame][3*i+1] + radius * np.sin( theta )    
            ax.fill(a, b, color='red', alpha=0.3)

        if (heron_p.CBF>0) and (heron_p.CBF_plot==1):
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

                        if (heron_p.CBF==1):
                            B = np.sqrt((x - obs_pos[frame][3*i+0]) ** 2 + (y - obs_pos[frame][3*i+1]) ** 2) - obs_pos[frame][3*i+2]
                            Bdot = ((x - obs_pos[frame][3*i+0]) * v * np.cos(theta) + (y - obs_pos[frame][3*i+1]) * v * np.sin(theta)) / np.sqrt((x - obs_pos[frame][3*i+0]) ** 2 + (y - obs_pos[frame][3*i+1]) ** 2)
                            hk[j, k] = Bdot + heron_p.gamma1/obs_pos[frame][3*i+2] * B

                        elif (heron_p.CBF==2):
                            R = v/heron_p.rmax*heron_p.gamma_TC1
                            B1 = np.sqrt( (obs_pos[frame][3*i+0]-x-R*np.cos(theta-np.pi/2))**2 + (obs_pos[frame][3*i+1]-y-R*np.sin(theta-np.pi/2))**2) - (obs_pos[frame][3*i+2]+R)
                            B2 = np.sqrt( (obs_pos[frame][3*i+0]-x-R*np.cos(theta+np.pi/2))**2 + (obs_pos[frame][3*i+1]-y-R*np.sin(theta+np.pi/2))**2) - (obs_pos[frame][3*i+2]+R)
                            hk[j, k] = np.max((B1,B2))

                # Plotting
                ax.contourf(X, Y, hk, levels=[-np.inf, 0], colors=['red'], alpha=0.1)
        


        ax.axis([-15, 15, -15, 15])  # Adjust these limits as needed
        ax.axis([-15, 15, -15, 15])  # Adjust these limits as needed
    

        # ax.autoscale(enable=True, axis='x', tight=True)
        # ax.autoscale(enable=True, axis='y', tight=True)
    
    frames = range(0, len(states), plot_iter)
    anim = FuncAnimation(fig, update, frames, repeat=False)
    plt.show()


def plot_inputs(t, reference, states, inputs, Fmax):
    FS = 18
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Plot the figure-eight trajectory
    axs[0, 0].plot(t, states[0:-1,3], 'k', linewidth=2) 
    axs[0, 0].plot(t, reference[0:len(t),3], 'b--', alpha=0.7)
    axs[0, 0].set_xlabel("Time", fontsize=FS)
    axs[0, 0].set_ylabel("Speed [m/s]", fontsize=FS)
    axs[0, 0].grid(True)
    axs[0, 0].autoscale(enable=True, axis='x', tight=True)
    axs[0, 0].autoscale(enable=True, axis='y', tight=True)

    # Plot the headings
    axs[0, 1].plot(t, states[0:-1,4], 'k', linewidth=2) 
    axs[0, 1].plot(t, reference[0:len(t),4], 'b--', alpha=0.7)
    axs[0, 1].set_xlabel("Time", fontsize=FS)
    axs[0, 1].set_ylabel("Rot. Speed [rad/s]", fontsize=FS)
    axs[0, 1].grid(True)
    axs[0, 1].autoscale(enable=True, axis='x', tight=True)
    axs[0, 1].autoscale(enable=True, axis='y', tight=True)

    # Plot the figure-eight trajectory
    axs[1, 0].plot(t, states[0:-1,5], 'k')
    axs[1, 0].plot(t, states[0:-1,5]*0 + Fmax, 'r--')
    axs[1, 0].plot(t, states[0:-1,5]*0 - Fmax, 'r--')
    axs[1, 0].set_xlabel("Time", fontsize=FS)
    axs[1, 0].set_ylabel("Left Thrust", fontsize=FS)
    axs[1, 0].grid(True)
    axs[1, 0].autoscale(enable=True, axis='x', tight=True)
    axs[1, 0].set_ylim(-Fmax-1, Fmax+1)

    # Plot the headings
    axs[1, 1].plot(t, states[0:-1,6], 'k')
    axs[1, 1].plot(t, states[0:-1,6]*0 + Fmax, 'r--')
    axs[1, 1].plot(t, states[0:-1,6]*0 - Fmax, 'r--')
    axs[1, 1].set_xlabel("Time", fontsize=FS)
    axs[1, 1].set_ylabel("Right Thrust", fontsize=FS)
    axs[1, 1].grid(True)
    axs[1, 1].autoscale(enable=True, axis='x', tight=True)
    axs[1, 1].set_ylim(-Fmax-1, Fmax+1)

    plt.tight_layout()
    plt.show()
