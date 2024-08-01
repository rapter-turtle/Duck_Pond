import numpy as np

class load_ship_param:
    # Heron
    M = 36*1.5 # Mass [kg]
    I = 8.35*1.5 # Inertial tensor [kg m^2]
    L = 0.73 # length [m]
    radius = L*1.5
    Xu = 0
    Xuu = 16.9 # N/(m/s)^2
    Yv = 10
    Nr = 0
    Nrr = 139/2 # Nm/(rad/s)^2
    
    vmin = 0.4
    vmax = 1.2
    
    Fxmax = 15
    Fxmin = -5
    dFxmax = 5

    # MPC param
    dt = 0.1
    N = 10

    Q = 2*np.diag([0, 1, 5, 0, 100, 15, 1e-2, 1e-2])
    R = 2*np.diag([1e-1,  # N=20
                   1e-1]) # N=20
    
    # CBF param
    CBF = 2 # 0-DC 1-ED 2-TC   
    CBF_plot = 1 # 0-plot off / 1-Plot on

    #### CBF = 1 ####
    gamma1 = 1
    gamma2 = 0.1

    #### CBF = 2 ####
    TCCBF = 3 # 1-B1(right avoid), 2-B2, 3-B1+B2
    rmax = 0.2
    gamma_TC1 = 1
    gamma_TC2 = 0.1

    
    def __init__(self): 
        print(1)   
    