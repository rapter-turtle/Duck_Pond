import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



def animateASV_recovery(states, inputs, tship, mpc_result, con_pos, t, Fmax, disturb, CBF):
    # Define the geometry of the twin-hull ASV
    hullLength = 0.7  # Length of the hull
    hullWidth = 0.2   # Width of each hull
    separation = 0.4  # Distance between the two hulls
    bodyLength = hullLength  # Length of the body connecting the hulls
    bodyWidth = 0.25  # Width of the body connecting the hulls

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4)
    
    
    ax_big = fig.add_subplot(gs[:, 0:2])
    ax_speed = fig.add_subplot(gs[0, 2])
    ax_rot_speed = fig.add_subplot(gs[0, 3])
    ax_thrust = fig.add_subplot(gs[1, 2])
    ax_disturb = fig.add_subplot(gs[1, 3])

    def update(frame):
        position = states[frame,0:2]
        heading = states[frame,2]
        # print(inputs[frame,0:2])
        force_left = inputs[frame,0]/100
        force_right = inputs[frame,1]/100

        # Clear the previous plots
        ax_big.clear()
        ax_speed.clear()
        ax_rot_speed.clear()
        ax_thrust.clear()
        ax_disturb.clear()
                
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
            ax_big.arrow(force_left_position[0], force_left_position[1], force_left*np.cos(heading), force_left*np.sin(heading), head_width=0.1, head_length=0.1, fc='b', ec='b')
        else:
            ax_big.arrow(force_left_position[0], force_left_position[1], force_left*np.cos(heading), force_left*np.sin(heading), head_width=0.1, head_length=0.1, fc='r', ec='r')
        
        if force_right > 0:
            ax_big.arrow(force_right_position[0], force_right_position[1], force_right*np.cos(heading), force_right*np.sin(heading), head_width=0.1, head_length=0.1, fc='b', ec='b')
        else:
            ax_big.arrow(force_right_position[0], force_right_position[1], force_right*np.cos(heading), force_right*np.sin(heading), head_width=0.1, head_length=0.1, fc='r', ec='r')

        # Plot the ASV
        ax_big.fill(hull1[0, :], hull1[1, :], 'b', alpha=0.3)
        ax_big.fill(hull2[0, :], hull2[1, :], 'b', alpha=0.3)
        ax_big.fill(body[0, :], body[1, :], 'b', alpha=0.3)
        ax_big.plot(position[0], position[1], 'go')  # Mark the position with a red dot
        ax_big.arrow(position[0], position[1], direction[0], direction[1], head_width=0.1, head_length=0.1, fc='k', ec='k')
        ax_big.axis('equal')

        ax_big.plot(mpc_result[frame][:,0], mpc_result[frame][:,1], 'c.')
        ax_big.plot(tship[0:frame,0], tship[0:frame,1], 'r--')  # Mark the position with a red dot
        dx = 10 / 2
        dy = 4 / 2

        # 회전 행렬 생성
        rotation_matrix = np.array([
            [np.cos(-tship[frame,2]), -np.sin(-tship[frame,2])],
            [np.sin(-tship[frame,2]), np.cos(-tship[frame,2])]
        ])

        # 사각형의 원래 꼭지점 좌표
        rectangle_coords = np.array([
            [-dx, -dy],
            [dx, -dy],
            [dx*1.2, 0],
            [dx, dy],
            [-dx, dy],
            [-dx, -dy]
        ])



        oa = con_pos[0]
        ob = con_pos[1]
        oc = con_pos[2]

        oxx = np.linspace( tship[frame,0]-10 , tship[frame,0]+5 , 20 )
        oyy = (-oa*oxx-oc)/ob
        ax_big.plot(oxx, oyy, 'k--', linewidth=2)

        oxx = np.linspace( tship[frame,0]-0.3 , tship[frame,0] + 0.2 , 20 )
        oyy = (-oa*oxx-oc)/ob
        ax_big.plot(oxx, oyy, 'c', linewidth=40, alpha=0.3)

        # 회전 및 이동 적용
        rotated_coords = np.dot(rectangle_coords, rotation_matrix)
        rotated_coords[:, 0] += tship[frame,0]
        rotated_coords[:, 1] += tship[frame,1]
        ax_big.plot(rotated_coords[:, 0], rotated_coords[:, 1], 'b-')
        ax_big.fill(rotated_coords[:, 0], rotated_coords[:, 1], 'blue', alpha=0.2)
        ax_big.scatter(tship[frame,0], tship[frame,1], color='red')  # 중심점 표시
        # ax_big.set_xlim(tship[frame,0]-15, tship[frame,0]+8)  # Adjust these limits as needed
        # ax_big.set_ylim(tship[frame,1]-10, tship[frame,1]+5)  # Adjust these limits as needed
        ax_big.set_xlim(-1, 1)  # Adjust these limits as needed
        ax_big.set_ylim(-1, 1)  # Adjust these limits as needed        
        ax_big.grid(True)
        FS = 14

        ax_big.set_xlabel("x [m]", fontsize=FS)
        ax_big.set_ylabel("y [m]", fontsize=FS)
        # ax_big.autoscale(enable=True, axis='x', tight=True)
        # ax_big.autoscale(enable=True, axis='y', tight=True)


        dist = 2.5
 

        # Plot the figure-eight trajectory
        # ax_speed.plot(t[0:frame], states[0:frame,3], 'k', linewidth=2) 
        # ax_speed.set_xlabel("Time", fontsize=FS)
        # ax_speed.set_title("Speed [m/s]", fontsize=FS)
        # ax_speed.grid(True)
        # ax_speed.autoscale(enable=True, axis='x', tight=True)
        # ax_speed.autoscale(enable=True, axis='y', tight=True)

        # Plot Rel X
        # ax_speed.plot(t[0:frame], states[0:frame,0] - tship[0:frame,0] - dist*np.sin(tship[0:frame,2]), 'k', linewidth=2) 
        # ax_speed.set_xlabel("Time", fontsize=FS)
        # ax_speed.set_title("Rel X", fontsize=FS)
        # ax_speed.grid(True)
        # ax_speed.autoscale(enable=True, axis='x', tight=True)
        # ax_speed.autoscale(enable=True, axis='y', tight=True)   

        # Plot CBF 1    
        ax_speed.plot(t[0:frame], (states[0:frame,0] + np.cos(states[0:frame,2])), 'k', linewidth=2, label='X error') 
        # ax_speed.plot(t[0:frame], states[0:frame,0], 'r', linewidth=2, label='Own ship X') 
        ax_speed.set_xlabel("Time", fontsize=FS)
        ax_speed.set_title("X", fontsize=FS)
        ax_speed.grid(True)
        ax_speed.autoscale(enable=True, axis='x', tight=True)
        ax_speed.autoscale(enable=True, axis='y', tight=True)    
        ax_speed.set_ylim(-2, 2)        



        # Plot the headings
        # ax_rot_speed.plot(t[0:frame], states[0:frame,4], 'k', linewidth=2) 
        # ax_rot_speed.set_xlabel("Time", fontsize=FS)
        # ax_rot_speed.set_title("Rot. Speed [rad/s]", fontsize=FS)
        # ax_rot_speed.grid(True)
        # ax_rot_speed.autoscale(enable=True, axis='x', tight=True)
        # ax_rot_speed.autoscale(enable=True, axis='y', tight=True)

        # Plot Rel Y
        # ax_rot_speed.plot(t[0:frame], states[0:frame,1] - tship[0:frame,1] + dist*np.cos(tship[0:frame,2]), 'k', linewidth=2) 
        # ax_rot_speed.set_xlabel("Time", fontsize=FS)
        # ax_rot_speed.set_title("Rel Y", fontsize=FS)
        # ax_rot_speed.grid(True)
        # ax_rot_speed.autoscale(enable=True, axis='x', tight=True)
        # ax_rot_speed.autoscale(enable=True, axis='y', tight=True)     

        # Plot CBF 2
        ax_rot_speed.plot(t[0:frame], (states[0:frame,1] + np.sin(states[0:frame,2])), 'k', linewidth=2, label='Y error') 
        # ax_rot_speed.plot(t[0:frame], states[0:frame,1], 'r', linewidth=2, label='Own ship Y') 
        ax_rot_speed.set_xlabel("Time", fontsize=FS)
        ax_rot_speed.set_title("Y", fontsize=FS)
        ax_rot_speed.grid(True)
        ax_rot_speed.legend()
        ax_rot_speed.autoscale(enable=True, axis='x', tight=True)
        ax_rot_speed.autoscale(enable=True, axis='y', tight=True)  
        ax_rot_speed.set_ylim(-2, 2)           



        # Plot the figure-eight trajectory
        ax_thrust.plot(t[0:frame], inputs[0:frame,0], 'g--', label='Right (MPC)')
        ax_thrust.plot(t[0:frame], inputs[0:frame,1], 'b--', label='Left (MPC)')
        
        ax_thrust.plot(t[0:frame], inputs[0:frame,0] + disturb[0:frame,3], 'g', label='Right (MPC+L1)')
        ax_thrust.plot(t[0:frame], inputs[0:frame,1] + disturb[0:frame,4], 'b', label='Left (MPC+L1)')
        
        ax_thrust.plot(t[0:frame], states[0:frame,5]*0 + Fmax, 'r--')
        ax_thrust.plot(t[0:frame], states[0:frame,5]*0 - Fmax, 'r--')
        ax_thrust.set_xlabel("Time", fontsize=FS)
        ax_thrust.set_title("Thrust", fontsize=FS)
        ax_thrust.grid(True)
        ax_thrust.legend()
        ax_thrust.autoscale(enable=True, axis='x', tight=True)
        ax_thrust.set_ylim(-Fmax-1, Fmax+1)

        ax_disturb.plot(t[0:frame], disturb[0:frame,0], label='(-) Disturb. X')
        ax_disturb.plot(t[0:frame], disturb[0:frame,1], label='(-) Disturb. Y')
        ax_disturb.plot(t[0:frame], disturb[0:frame,5], label='L1 X')
        ax_disturb.plot(t[0:frame], disturb[0:frame,6], label='L1 Y')
        ax_disturb.set_xlabel("Time", fontsize=FS)
        ax_disturb.set_title("Disturbance vs DOB", fontsize=FS)
        ax_disturb.grid(True)
        ax_disturb.legend()
        ax_disturb.autoscale(enable=True, axis='x', tight=True)

        # # ax_disturb.plot(t[0:frame], disturb[0:frame,4]+disturb[0:frame,0], label='(-) L1 Error. X')
        # # ax_disturb.plot(t[0:frame], disturb[0:frame,5]+disturb[0:frame,1], label='(-) L1 Error. N')
        # ax_disturb.plot(t[0:frame], disturb[0:frame,6]+disturb[0:frame,0], label='(-) DOB Error. X')
        # ax_disturb.plot(t[0:frame], disturb[0:frame,7]+disturb[0:frame,1], label='(-) DOB Error. N')        
        # ax_disturb.set_xlabel("Time", fontsize=FS)
        # ax_disturb.set_title("Disturbance vs DOB", fontsize=FS)
        # ax_disturb.grid(True)
        # ax_disturb.legend()
        # ax_disturb.autoscale(enable=True, axis='x', tight=True)
       

    anim = FuncAnimation(fig, update, frames=len(states), repeat=False)
    plt.show()




def animateASV(states, inputs, ref, yref, mpc_result, obs_pos):
    # Define the geometry of the twin-hull ASV
    hullLength = 0.7  # Length of the hull
    hullWidth = 0.2   # Width of each hull
    separation = 0.4  # Distance between the two hulls
    bodyLength = hullLength  # Length of the body connecting the hulls
    bodyWidth = 0.25  # Width of the body connecting the hulls

    fig, ax = plt.subplots()

    def update(frame):
        position = states[frame,0:2]
        heading = states[frame,2]
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


        theta = np.linspace( 0 , 2 * np.pi , 150 )
        
        for i in range(5):
            radius = obs_pos[frame][3*i+2]        
            a = obs_pos[frame][3*i+0] + radius * np.cos( theta )
            b = obs_pos[frame][3*i+1] + radius * np.sin( theta )    
            ax.plot(a, b)

        ax.set_xlim(-8, 8)  # Adjust these limits as needed
        ax.set_ylim(-6, 6)  # Adjust these limits as needed
        # ax.autoscale(enable=True, axis='x', tight=True)
        # ax.autoscale(enable=True, axis='y', tight=True)

    anim = FuncAnimation(fig, update, frames=len(states), repeat=False)
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
