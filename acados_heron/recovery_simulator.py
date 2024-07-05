from plot_asv import *
from gen_ref import *
from acados_setting import *

def recover_simulator(ship, tship, control_input, dt, disturbance, extra_control):

    M = 36 # Mass [kg]
    I = 8.35 # Inertial tensor [kg m^2]
    L = 0.73 # length [m]
    Xu = 10
    Xuu = 16.9 # N/(m/s)^2
    Nr = 5
    Nrr = 13 # Nm/(rad/s)^2


    # set up states & controls
    psi  = ship[2]
    v    = ship[3]
    r    = ship[4]
    n1  = ship[5]
    n2  = ship[6]

    n1d  = control_input[0]
    n2d  = control_input[1]

    n1_extra = extra_control[0]
    n2_extra = extra_control[1]

    # dynamics
    xdot = np.array([v*np.cos(psi),
                     v*np.sin(psi),
                     r,
                     ((n1+n2)-(Xu + Xuu*np.sqrt(v*v))*v + disturbance[0] + (n1_extra + n2_extra))/M ,
                     ((-n1+n2)*L/2 - (Nr + Nrr*np.sqrt(r*r))*r + disturbance[1] + (-n1_extra + n2_extra)*L/2)/I,
                     n1d,
                     n2d
                     ])
    # x_tship = np.array([10.0, 10.0, 0.1, 1]) # x,y,psi,u

    ship = xdot*dt + ship



    tpsi  = tship[2]
    tu    = tship[3]

    tship_xdot = np.array([tu*np.cos(tpsi),
                     tu*np.sin(tpsi),
                     0.0,
                     0.0
                     ])


    tship = tship_xdot*dt + tship

    return ship, tship
