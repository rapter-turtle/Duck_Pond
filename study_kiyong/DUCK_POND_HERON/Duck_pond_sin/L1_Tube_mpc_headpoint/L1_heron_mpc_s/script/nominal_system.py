import numpy as np
import math 

def nom_sym(state, dt, nstate):

 
    M = 37.758 # Mass [kg]
    I = 18.35 # Inertial tensor [kg m^2]    
    Xu = 8.9149
    Xuu = 11.2101
    Nr = 16.9542
    Nrrr = 12.8966
    Yv = 15
    Yvv = 3
    Yr = 6
    Nv = 6
    dist = 0.3 # 30cm
    head_dist = 1.0

    
    # set up states & controls

    psi  = nstate[2]
    u    = nstate[3]
    v    = nstate[4]
    r    = nstate[5]
    n1  = state[6]
    n2  = state[7]

    eps = 0.0001
    xdot = np.array([
        u * np.cos(psi) - v * np.sin(psi) - r*np.sin(psi),
        u * np.sin(psi) + v * np.cos(psi) + r*np.cos(psi),
        r,
        ((n1 + n2) - (Xu + Xuu * np.sqrt(u * u + eps)) * u) / M,
        ( -Yv * v - Yvv * np.sqrt(v * v + eps) * v - Yr * r) / M,
        ((-n1 + n2) * dist - (Nr + Nrrr * r * r) * r - Nv * v) / I,
        0.0,
        0.0
    ])


    x_t_nstate = xdot*dt + nstate
    
    x_t_nstate[6] = n1
    x_t_nstate[7] = n2

    nom_state = np.array([nstate[0],
                          nstate[1],
                          nstate[3]*np.cos(nstate[2]) - nstate[4]*np.sin(nstate[2]) - head_dist*nstate[5]*np.sin(nstate[2]),
                          nstate[3]*np.sin(nstate[2]) + nstate[4]*np.cos(nstate[2]) + head_dist*nstate[5]*np.cos(nstate[2])])

    real_state = np.array([state[0] + np.cos(state[2]),
                           state[1] + np.sin(state[2]),
                           state[3]*np.cos(state[2]) - state[4]*np.sin(state[2]) - head_dist*state[5]*np.sin(state[2]),
                           state[3]*np.sin(state[2]) + state[4]*np.cos(state[2]) + head_dist*state[5]*np.cos(state[2])
                          ])


    state_error = real_state - nom_state

    feedback = 0.001*np.array([-622.0*state_error[0] - 6.4873*state_error[2],
                             -622.0*state_error[1] - 6.4873*state_error[3]
    ])
    print(feedback)

    # feedback = 0.0001*np.array([0.0,
    #                             0.0])

    feedback_con = np.array([0.5*(M*np.cos(state[2]) + I*np.sin(state[2])/(head_dist*dist))*feedback[0] + 0.5*(M*np.sin(state[2]) - I*np.cos(state[2])/(head_dist*dist))*feedback[1],
                             0.5*(M*np.cos(state[2]) - I*np.sin(state[2])/(head_dist*dist))*feedback[0] + 0.5*(M*np.sin(state[2]) + I*np.cos(state[2])/(head_dist*dist))*feedback[1]
                            ])  


    return x_t_nstate, feedback_con, feedback




