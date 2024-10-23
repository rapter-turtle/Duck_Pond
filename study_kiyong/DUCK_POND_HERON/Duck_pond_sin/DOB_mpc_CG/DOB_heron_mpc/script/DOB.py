import numpy as np
import math 

def DOB(state, state_estim, param_filtered, dt, w_cutoff):

 
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

    psi  = state[2]
    u    = state[3]
    v    = state[4]
    r    = state[5]
    n1  = state[6]
    n2  = state[7]


    f_usv = np.array([[u*np.cos(psi) - v*np.sin(psi)],
                      [u*np.sin(psi) + v*np.cos(psi)],
                      [r],
                      [((n1 + n2) - (Xu + Xuu * np.sqrt(u * u)) * u) / M],
                      [( -Yv * v - Yvv * np.sqrt(v * v ) * v - Yr * r) / M],
                      [((-n1 + n2) * dist - (Nr + Nrrr * r * r) * r - Nv * v) / I]
          ])

    p = np.array([M*u*np.cos(psi) - M*v*np.sin(psi),
                  M*u*np.sin(psi) + M*v*np.cos(psi),
                  I*r
    ])

    l = -1*np.array([[0, 0, -M*u*np.sin(psi)-M*v*np.cos(psi), M*np.cos(psi),-M*np.sin(psi), 0],
                  [0, 0, M*u*np.cos(psi)-M*v*np.sin(psi), M*np.sin(psi),M*np.cos(psi), 0],
                  [0, 0, 0, 0, 0, I]
                  ])


    g2 = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [np.cos(psi)/M, np.sin(psi)/M, 0],
                   [-np.sin(psi)/M, np.cos(psi)/M, 0],
                   [0, 0, 1/I] 
    ])

    # print(g2@state_estim.reshape(3, 1))
    # print(g2@p.reshape(3, 1))
    # print(f_usv)
    # print(g2@state_estim.reshape(3, 1) + g2@p.reshape(3, 1) + f_usv)
    # print(l)
    # print(g2)
    xdot = l@(g2@state_estim.reshape(3, 1) + g2@p.reshape(3, 1) + f_usv)

    x_t_plus = xdot*dt + state_estim.reshape(3, 1)
    
    param_filtered = x_t_plus + p.reshape(3, 1)
    # print(p)
    return x_t_plus, param_filtered




