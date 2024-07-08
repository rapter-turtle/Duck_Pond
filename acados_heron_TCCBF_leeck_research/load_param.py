import numpy as np

class load_param:
    # Heron
    M = 36 # Mass [kg]
    I = 8.35 # Inertial tensor [kg m^2]
    L = 0.73 # length [m]
    Xu = 10
    Xuu = 16.9 # N/(m/s)^2
    Nr = 5
    Nrr = 13 # Nm/(rad/s)^2
    Fmax = 45

    # MPC param
    dt = 0.2
    N = 10
    Q = 2*np.diag([1, 1, 10, 50, 1, 1e-6, 1e-6])
    R = 2*np.diag([1e-6, 
                   1e-6])

    # CBF param
    CBF = 1 # 0-DC 1-ED 2-TC   
    #### CBF = 1 ####
    gamma1 = 0.75
    gamma2 = 0.3

    #### CBF = 2 ####
    rmax = 0.6
    gamma_TC1 = 2
    gamma_TC2 = 0.4
    
    def __init__(self): 
        print(1)   
    