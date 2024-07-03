
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math

def plotTrackProj(rel,simX, Nsim, t):
    # load track

    v=t

    l = 3.5
    # state[3] + l*math.cos(state[5]) - state[8]
    # state[4] + l*math.sin(state[5]) - state[9]

    # x = rel[:,0] + simX[:,8]
    # y = rel[:,1] + simX[:,9]
    # plot racetrack map
    x = simX[:,8]
    y = simX[:,9]


    # #Setup plot
    plt.figure(figsize=(10, 5))
    plt.ylim(bottom=-50,top=50)
    plt.xlim(left=0,right=250)
    plt.ylabel('y[m]')
    plt.xlabel('x[m]')

    # Draw driven trajectory
    heatmap = plt.scatter(x,y, c='r',s=0.3, edgecolor='none', marker='o',linewidth=0.5)
    cbar = plt.colorbar(heatmap, fraction=0.035)
    cbar.set_label("Time [s]")
    ax = plt.gca()
    # ax.set_aspect('equal', 'box')
    ax.grid(True)

    N = 20
    gap = int(Nsim/N)
    for i in range(20):
        print(i*gap)
        current_x = simX[i*gap,3]
        current_y = simX[i*gap,4]
        current_psi = simX[i*gap,5]

        # print("x",current_x)
        # print("y",current_y)
        # print("psi",current_psi)

        # Plot ship's shape polygon according to the current state
        ship_length = 3.5  # Example length of the ship
        ship_width = 1  # Example width of the ship

        # Define ship shape vertices
        ship_vertices = np.array([[current_x - 0.5 * ship_length * np.cos(current_psi) - 0.5 * ship_width * np.sin(current_psi),
                                    current_y - 0.5 * ship_length * np.sin(current_psi) + 0.5 * ship_width * np.cos(current_psi)],
                                [current_x + 0.5 * ship_length * np.cos(current_psi) - 0.5 * ship_width * np.sin(current_psi),
                                    current_y + 0.5 * ship_length * np.sin(current_psi) + 0.5 * ship_width * np.cos(current_psi)],
                                [current_x + 0.8 * ship_length * np.cos(current_psi),
                                    current_y + 0.8 * ship_length * np.sin(current_psi)], 
                                [current_x + 0.5 * ship_length * np.cos(current_psi) + 0.5 * ship_width * np.sin(current_psi),
                                    current_y + 0.5 * ship_length * np.sin(current_psi) - 0.5 * ship_width * np.cos(current_psi)],
                                [current_x - 0.5 * ship_length * np.cos(current_psi) + 0.5 * ship_width * np.sin(current_psi),
                                    current_y - 0.5 * ship_length * np.sin(current_psi) - 0.5 * ship_width * np.cos(current_psi)]])
        

        # Plot ship's shape polygon
        ship_polygon = Polygon(ship_vertices, closed=True, edgecolor='b', facecolor='b')


        ax.add_patch(ship_polygon) 

    plt.axes().set_aspect('equal')


def plotRes(simX,simU,t, sim_l1_con, sim_param, sim_x_estim, real, sim_filtered, dis, sim_l1_only_con):
    # plot results
    N = 5
    M = 2
    plt.figure()
    plt.subplot(N, M, 1)
    plt.step(t,sim_x_estim[:,3] - real[:,3] , color='r')
    plt.ylabel('estim error Vy')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 2)
    plt.plot(t, sim_x_estim[:,2] - real[:,2])
    plt.ylabel('estim error Vx')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 3)
    plt.plot(t, real[:,1])
    # plt.plot(t, sim_x_estim[:,1])
    plt.ylabel('y distance ')
    # plt.ylim(bottom=-1,top=1)
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 4)
    plt.plot(t, real[:,0])
    plt.ylabel('x distance ')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 5)
    plt.plot(t, sim_param[:,0]-sim_filtered[:,0] )
    # plt.plot(t, sim_filtered[:,0])
    plt.ylabel('param x ')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 6)
    plt.plot(t, sim_param[:,1] - sim_filtered[:,1])
    # plt.plot(t, sim_filtered[:,1])
    plt.ylabel('param y ')
    plt.xlabel('t')
    plt.grid(True)    
    plt.subplot(N, M, 7)
    plt.plot(t, sim_param[:,2])
    plt.ylabel('param psi ')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 8)
    plt.plot(t, sim_l1_con[:,1] - simX[:,7])
    plt.plot(t, dis[:,1])
    plt.ylabel('l1 y ')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 9)
    plt.plot(t, sim_l1_con[:,0]- simX[:,8])
    plt.plot(t, -dis[:,0])
    plt.ylabel('l1 x ')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(N, M, 10)
    plt.plot(t, simX[:,7] )
    plt.ylabel('Tau y mpc')
    plt.xlabel('t')
    plt.grid(True)       



def current_plot(x, ax, t, state):
    l = 3.5

    current_x = state[3]
    current_y = state[4]
    current_psi = state[5]

    Target_current_x = state[8]
    Target_current_y = state[9]
    Target_current_psi = 0.0

    # Plot ship's shape polygon according to the current state
    ship_length = 3.5  # Example length of the ship
    ship_width = 1  # Example width of the ship

    # Define ship shape vertices
    ship_vertices = np.array([[current_x - 0.5 * ship_length * np.cos(current_psi) - 0.5 * ship_width * np.sin(current_psi),
                                current_y - 0.5 * ship_length * np.sin(current_psi) + 0.5 * ship_width * np.cos(current_psi)],
                               [current_x + 0.5 * ship_length * np.cos(current_psi) - 0.5 * ship_width * np.sin(current_psi),
                                current_y + 0.5 * ship_length * np.sin(current_psi) + 0.5 * ship_width * np.cos(current_psi)],
                               [current_x + 0.8 * ship_length * np.cos(current_psi),
                                current_y + 0.8 * ship_length * np.sin(current_psi)], 
                               [current_x + 0.5 * ship_length * np.cos(current_psi) + 0.5 * ship_width * np.sin(current_psi),
                                current_y + 0.5 * ship_length * np.sin(current_psi) - 0.5 * ship_width * np.cos(current_psi)],
                               [current_x - 0.5 * ship_length * np.cos(current_psi) + 0.5 * ship_width * np.sin(current_psi),
                                current_y - 0.5 * ship_length * np.sin(current_psi) - 0.5 * ship_width * np.cos(current_psi)]])


    Target_ship_vertices = np.array([[Target_current_x - 0.5 * ship_length * np.cos(Target_current_psi) - 0.5 * ship_width * np.sin(Target_current_psi),
                                Target_current_y - 0.5 * ship_length * np.sin(Target_current_psi) + 0.5 * ship_width * np.cos(Target_current_psi)],
                               [Target_current_x + 0.5 * ship_length * np.cos(Target_current_psi) - 0.5 * ship_width * np.sin(Target_current_psi),
                                Target_current_y + 0.5 * ship_length * np.sin(Target_current_psi) + 0.5 * ship_width * np.cos(Target_current_psi)],
                               [Target_current_x + 0.8 * ship_length * np.cos(Target_current_psi),
                                Target_current_y + 0.8 * ship_length * np.sin(Target_current_psi)], 
                               [Target_current_x + 0.5 * ship_length * np.cos(Target_current_psi) + 0.5 * ship_width * np.sin(Target_current_psi),
                                Target_current_y + 0.5 * ship_length * np.sin(Target_current_psi) - 0.5 * ship_width * np.cos(Target_current_psi)],
                               [Target_current_x - 0.5 * ship_length * np.cos(Target_current_psi) + 0.5 * ship_width * np.sin(Target_current_psi),
                                Target_current_y - 0.5 * ship_length * np.sin(Target_current_psi) - 0.5 * ship_width * np.cos(Target_current_psi)]])


    # Plot ship's shape polygon
    ship_polygon = Polygon(ship_vertices, closed=True, edgecolor='b', facecolor='b')
    ax.add_patch(ship_polygon)
    Target_ship_polygon = Polygon(Target_ship_vertices, closed=True, edgecolor='r', facecolor='r')
    ax.add_patch(Target_ship_polygon)

    # Plot the trajectory of (x, y) for the prediction horizon
    # predicted_horizon_x = [sub_list[3] for sub_list in x]
    # predicted_horizon_y = [sub_list[4] for sub_list in x]
    # ax.plot(predicted_horizon_x -np.cos(state[5]) + state[8], predicted_horizon_y -np.sin(state[5])  + state[9], 'r-', label='Predicted Horizon')
    # Add labels and legend
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.legend()
    # ax.axis('equal')
    ax.text(0.5, 1.02, f'Time: {t}', fontsize=12, ha='center', va='bottom', transform=ax.transAxes)
    # plt.figure(figsize=(10, 4))
    # plt.figure(figsize=(20, 2))
    plt.axis('tight')
    plt.axis('equal')
    
    plt.ylim(bottom=-20,top=20)
    plt.xlim(left=-2,right=180)
    
    plt.draw() 
    plt.pause(0.001)

    
    ax.clear() 
    

