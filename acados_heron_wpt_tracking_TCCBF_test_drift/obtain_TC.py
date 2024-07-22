from plot_ship import *
from acados_setting import *
from load_param import *
from ship_integrator import *

def main(target_speed):
    ship_p = load_ship_param
    dt = 0.2
    Tf = 25
    Nsim = int(Tf/dt)
    Fx_init = ship_p.Xu*target_speed + ship_p.Xuu*target_speed**2
    x0 = np.array([0.0 , # x
                    0.0, # y
                    0.0, # psi
                    target_speed, # surge
                    0.0, # sway
                    0.0, # rot-vel
                    Fx_init,  # Fx
                    0])  # Fn    

    simX = np.zeros((Nsim+1, 8))
    simU = np.zeros((Nsim+1, 2))
    simX[0,:] = x0
    
    for i in range(Nsim):
        simU[i, :] = [ship_p.dFxmax, -ship_p.dFxmax]
        if np.abs(simX[i,6]-simX[i,7]) >= 20:
            simU[i, :] = [0, 0]
        simX[i+1, :] = ship_integrator(simX[i, :], simU[i,:], dt)


    simU[i+1, :] = simU[i, :]

    return simX, Nsim



def plot_TC(simX, Nsim):

    fig, axs = plt.subplots(2,2, figsize=(10,8))
    # Combine first two columns for the ASV plot
    ax_asv = axs[0, 0]
    # CBF animation setup
    ax_surge = axs[0, 1]
    ax_rot = axs[1, 0]
    ax_thrust = axs[1, 1]
    ship_p = load_ship_param
    dt = ship_p.dt

    # Define the geometry of the twin-hull ASV
    hullLength = 0.7  # Length of the hull
    hullWidth = 0.2   # Width of each hull
    separation = 0.4  # Distance between the two hulls
    bodyLength = hullLength  # Length of the body connecting the hulls
    bodyWidth = 0.25  # Width of the body connecting the hulls

    FS = 16

    for i in range(Nsim):
        if i%10 == 0:
            position = simX[i,0:2]
            heading = simX[i,2]
            
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

            # Plot the ASV
            ax_asv.fill(hull1[0, :], hull1[1, :], 'b', alpha=0.3)
            ax_asv.fill(hull2[0, :], hull2[1, :], 'b', alpha=0.3)
            ax_asv.fill(body[0, :], body[1, :], 'b', alpha=0.3)
                
            arrow_length = 0.5
            direction = np.array([np.cos(heading), np.sin(heading)]) * arrow_length        # Plot the ASV
            ax_asv.fill(body[0, :], body[1, :], 'b', alpha=0.3)
            ax_asv.arrow(position[0], position[1], direction[0], direction[1], head_width=0.1, head_length=0.1, fc='k', ec='k')       

    ax_asv.plot(simX[:,0], simX[:,1], '--')
    ax_asv.xaxis.label.set_size(FS)
    ax_asv.yaxis.label.set_size(FS)
    ax_asv.set_xlabel("x [m]", fontsize=FS)
    ax_asv.set_ylabel("y [m]", fontsize=FS)
    ax_asv.set_aspect('equal')
    ax_asv.set_aspect('equal')
    xyax = 15
    ax_asv.grid(True)
    # ax_asv.set(xlim=(-xyax,xyax),ylim=(-xyax,xyax))

    t = np.arange(0, dt*(Nsim+0.5), dt)
    ax_surge.plot(t, simX[:,3], linewidth=2, label='surge') 
    ax_surge.plot(t, simX[:,4], linewidth=2, label='sway') 
    ax_surge.plot(t, np.sqrt(simX[:,3]**2+simX[:,4]**2), linewidth=2, label='Speed') 
    ax_surge.set_xlabel("Time", fontsize=FS)
    ax_surge.set_ylabel("Surge [m/s]", fontsize=FS)
    ax_surge.grid(True)
    ax_surge.autoscale(enable=True, axis='x', tight=True)
    ax_surge.autoscale(enable=True, axis='y', tight=True)
    ax_surge.set_ylim(-2, 2)
    ax_surge.legend()

    ax_rot.plot(t, simX[:,5], linewidth=2) 
    ax_rot.set_xlabel("Time", fontsize=FS)
    ax_rot.set_ylabel("Rot. Speed [rad/s]", fontsize=FS)
    ax_rot.grid(True)
    ax_rot.autoscale(enable=True, axis='x', tight=True)
    ax_rot.autoscale(enable=True, axis='y', tight=True)
    ax_rot.set_ylim(-1, 1)

    # Plot the figure-eight trajectory
    ax_thrust.plot(t, simX[:,6], label='left')
    ax_thrust.plot(t, simX[:,7], label='right')
    ax_thrust.plot(t, simX[:,6]*0 + ship_p.Fxmax, 'r--')
    ax_thrust.plot(t, simX[:,6]*0 + ship_p.Fxmin, 'r--')
    ax_thrust.set_xlabel("Time", fontsize=FS)
    ax_thrust.set_ylabel("Thrust", fontsize=FS)
    ax_thrust.grid(True)
    ax_thrust.autoscale(enable=True, axis='x', tight=True)
    ax_thrust.set_ylim(ship_p.Fxmin-1, ship_p.Fxmax+1)
    ax_thrust.legend()

    fig.tight_layout()  # axes 사이 간격을 적당히 벌려줍니다.

if __name__ == '__main__':
    simX, Nsim = main(1.5)
    plot_TC(simX, Nsim)
    simX, Nsim = main(1.0)
    plot_TC(simX, Nsim)
    simX, Nsim = main(0.5)
    plot_TC(simX, Nsim)

    plt.show()
