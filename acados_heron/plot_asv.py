import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animateASV(positions, headings, inputs, ref, yref, mpc_result):
    # Define the geometry of the twin-hull ASV
    hullLength = 0.7  # Length of the hull
    hullWidth = 0.2   # Width of each hull
    separation = 0.4  # Distance between the two hulls
    bodyLength = hullLength  # Length of the body connecting the hulls
    bodyWidth = 0.25  # Width of the body connecting the hulls

    fig, ax = plt.subplots()

    def update(frame):
        position = positions[frame]
        heading = headings[frame]
        print(inputs[frame,0:2])
        print()
        force_left = inputs[frame,0]/100
        force_right = inputs[frame,1]/100
        # Clear the previous plot
        ax.clear()
        
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
        ax.plot(yref[frame,:,0], yref[frame,:,1], 'mo')
        ax.plot(mpc_result[frame][:,0], mpc_result[frame][:,1], 'c.')

        ax.set_xlim(-4, 4)  # Adjust these limits as needed
        ax.set_ylim(-4, 4)  # Adjust these limits as needed

    anim = FuncAnimation(fig, update, frames=len(positions), repeat=False)
    plt.show()
