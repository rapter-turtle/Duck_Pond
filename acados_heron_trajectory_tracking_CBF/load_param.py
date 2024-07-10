import numpy as np

class load_param:
    # Heron
    M = 36 # Mass [kg]
    I = 8.35*1 # Inertial tensor [kg m^2]
    L = 0.73 # length [m]
    Xu = 10
    Xuu = 16.9 # N/(m/s)^2
    Nr = 5*1
    Nrr = 13 # Nm/(rad/s)^2
    Fmax = 45
    dFmax = 45

    # MPC param
    dt = 0.2
    N = 20
    Q = 2*np.diag([1, 1, 0, 25, 5, 1e-7, 1e-7])
    R = 2*np.diag([1e-5, 
                   1e-5])

    # CBF param
    CBF_plot = 0
    CBF = 1 # 0-DC 1-ED 2-TC   
    #### CBF = 1 ####
    gamma1 = 0.75
    gamma2 = 0.4

    #### CBF = 2 ####
    rmax = 0.6
    gamma_TC1 = 4
    gamma_TC2 = 0.025
    
    def __init__(self): 
        print(1)   
    