from acados_setting import *
from load_param import *

def kinematic_integrator(vehicle, control_input, dt):
    # set up states & controls
    psi  = vehicle[2]
    u    = vehicle[3]
    rot  = vehicle[4]
    acc  = vehicle[5]

    drot  = control_input[0]
    dacc  = control_input[1]


    # dynamics
    xdot = np.array([u*np.cos(psi),
                     u*np.sin(psi),
                     rot,
                     acc,                     
                     drot,
                     dacc
                     ])

    vehicle = xdot*dt + vehicle

    return vehicle
