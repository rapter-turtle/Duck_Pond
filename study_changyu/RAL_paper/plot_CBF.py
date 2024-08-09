import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm, animation
from matplotlib.colors import Normalize
import time
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def main():
    
    fig, axs = plt.subplots(1,2,figsize=(17,5))
    # Combine first two columns for the ASV plot
    size = 1.5
    hullLength = 0.7 * size # Length of the hull
    hullWidth = 0.3 * size   # Width of each hull 

    position = [-5,0]
    heading = 0
    speed = 1.5
    theta = np.linspace( 0 , 2 * np.pi , 100 )
    ox = 2
    oy = 0
    orad =2 
    gamma1 = 0.5
    rmax = 0.3
    gamma_TC1 = 1
    radius = 0
        
    CBF = 1
    xmin=-8
    xmax=2
    ymin=-3.5
    ymax= 3.5
    # Define the grid for plotting
    x_range = np.linspace(xmin, xmax, 150)
    y_range = np.linspace(ymin, ymax, 150)

    X, Y = np.meshgrid(x_range, y_range)
    
    FS = 24

    ax_asv = axs[0]
    ax_asv2 = axs[1]
    # Clear the previous plot
    ax_asv.clear()
    ax_asv2.clear()
    
    hk = np.ones_like(X)*1000
    # Initialize hk array
    # Calculate hk for each point in the grid
    for j in range(len(x_range)):
        for k in range(len(y_range)):
            x = X[j, k]
            y = Y[j, k]
            head_ang = heading  # Assume theta = 0 for simplicity
            u = speed  # Assume constant speed for simplicity

            B = np.sqrt((x - ox) ** 2 + (y - oy) ** 2) - orad
            Bdot = ((x - ox) * (u * np.cos(head_ang)) + (y - oy) *(u * np.sin(head_ang))) / np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            hk[j, k] = np.min((hk[j, k], Bdot + gamma1 * B))

    ax_asv.imshow(hk, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap=cm.bone, aspect='auto', alpha=1)

    # ax_asv.contourf(X, Y, hk, levels=np.linspace(hk.min(),hk.max(),40), alpha=1, cmap=cm.bone)
    ax_asv.contourf(X, Y, hk, levels=[-0.015, 0.015], colors=['red'], alpha=0.88)

    cbf_min = np.min(hk)
    cbf_max = np.max(hk)
    for j in range(len(x_range)):
        for k in range(len(y_range)):
            if np.abs(np.arctan2((oy-y),(ox-x))-head_ang)>np.pi/2:
                hk[j, k] = cbf_max
                
    cbar_ax = fig.add_axes([0.46, 0.13, 0.015, 0.8])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap='bone', norm=mcolors.Normalize(vmin=cbf_min, vmax=cbf_max))
    sm.set_array([])
    cbar_hk = fig.colorbar(sm, ax=ax_asv, cax=cbar_ax)
    cbar_hk.ax.set_title("CBF", fontsize=FS-2)
    cbar_hk.ax.axhline(y=0, color='red', linewidth=2)  # Adjust the normalization accordingly
    cbar_hk.ax.tick_params(labelsize=FS-2)  # Set the font size for the colorbar tick labels

    
    hk = np.ones_like(X)*1000
    for j in range(len(x_range)):
        for k in range(len(y_range)):
            x = X[j, k]
            y = Y[j, k]
            head_ang = heading  # Assume theta = 0 for simplicity
            u = speed  # Assume constant speed for simplicity

            R = u/rmax*gamma_TC1
            B1 = np.sqrt( (ox-x-R*np.cos(head_ang-np.pi/2))**2 + (oy-y-R*np.sin(head_ang-np.pi/2))**2) - (orad+R)
            B2 = np.sqrt( (ox-x-R*np.cos(head_ang+np.pi/2))**2 + (oy-y-R*np.sin(head_ang+np.pi/2))**2) - (orad+R)
            hk[j, k] = np.min((hk[j, k], np.max((B1,B2))))
    ax_asv2.imshow(hk, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap=cm.bone, aspect='auto', alpha=1)
    # ax_asv2.contourf(X, Y, hk, levels=np.linspace(hk.min(),hk.max(),40), alpha=1, cmap=cm.bone)
    ax_asv2.contourf(X, Y, hk, levels=[-0.02, 0.02], colors=['red'], alpha=0.88)


    
    cbf_min = np.min(hk)
    cbf_max = np.max(hk)
    for j in range(len(x_range)):
        for k in range(len(y_range)):
            if np.abs(np.arctan2((oy-y),(ox-x))-head_ang)>np.pi/2:
                hk[j, k] = cbf_max
                

    cbar_ax = fig.add_axes([0.95, 0.13, 0.015, 0.8])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap='bone', norm=mcolors.Normalize(vmin=cbf_min, vmax=cbf_max))
    sm.set_array([])
    cbar_hk = fig.colorbar(sm, ax=ax_asv2, cax=cbar_ax)
    cbar_hk.ax.set_title("CBF", fontsize=FS-2)
    cbar_hk.ax.axhline(y=0, color='red', linewidth=2)  # Adjust the normalization accordingly
    cbar_hk.ax.tick_params(labelsize=FS-2)  # Set the font size for the colorbar tick labels

    radius = radius
    a = position[0] + radius * np.cos( theta )
    b = position[1] + radius * np.sin( theta )    
    ax_asv.fill(a, b, color='white', alpha=1)
    
    hull = np.array([[-hullLength/2, hullLength/2, hullLength/2+0.25, hullLength/2,-hullLength/2, -hullLength/2, -hullLength/2],
                    [hullWidth/2, hullWidth/2, 0, -hullWidth/2, -hullWidth/2, 0, hullWidth/2]])

    R = np.array([[np.cos(heading), -np.sin(heading)],
                [np.sin(heading), np.cos(heading)]])

    hull = R @ hull
    hull = hull + np.array(position).reshape(2, 1)
    arrow_length = 1
    direction = np.array([np.cos(heading), np.sin(heading)]) * arrow_length

    # Plot the ASV
    ax_asv.fill(hull[0, :], hull[1, :], 'w', alpha=0.9, edgecolor='black')
    ax_asv.arrow(position[0], position[1], direction[0], direction[1], head_width=0.2, head_length=0.2, fc='k', ec='k')       
    ax_asv2.fill(hull[0, :], hull[1, :], 'w', alpha=0.9, edgecolor='black')
    ax_asv2.arrow(position[0], position[1], direction[0], direction[1], head_width=0.2, head_length=0.2, fc='k', ec='k')       

    ax_asv.set_xlabel('x [m]', fontsize=FS)  # Set x-axis label and font size
    ax_asv.set_ylabel('y [m]', fontsize=FS)  # Set y-axis label and font size

    ax_asv.tick_params(axis='x', labelsize=FS)  # Set x-axis tick label size
    ax_asv.tick_params(axis='y', labelsize=FS)  # Set y-axis tick label size

    ax_asv2.set_xlabel('x [m]', fontsize=FS)  # Set x-axis label and font size
    ax_asv2.set_ylabel('y [m]', fontsize=FS)  # Set y-axis label and font size

    ax_asv2.tick_params(axis='x', labelsize=FS)  # Set x-axis tick label size
    ax_asv2.tick_params(axis='y', labelsize=FS)  # Set y-axis tick label size

    radius = orad - radius
    a = ox + radius * np.cos( theta )
    b = oy + radius * np.sin( theta )    
    ax_asv.fill(a, b, facecolor='w', alpha=1, edgecolor='black')
    ax_asv.fill(a, b, facecolor='k', alpha=0.1, edgecolor='black')


    ax_asv2.fill(a, b, facecolor='w', alpha=1, edgecolor='black')
    ax_asv2.fill(a, b, facecolor='k', alpha=0.15, edgecolor='black')


    ax_asv.set_aspect('equal')
    ax_asv.set(xlim=(xmin, xmax),ylim=(ymin, ymax))
    
    ax_asv2.set_aspect('equal')
    ax_asv2.set(xlim=(xmin, xmax),ylim=(ymin, ymax))

    fig.tight_layout()  # axes 사이 간격을 적당히 벌려줍니다.
    plt.savefig('cbf_compare.pdf')
    
    # plt.show()

if __name__ == '__main__':
    main()