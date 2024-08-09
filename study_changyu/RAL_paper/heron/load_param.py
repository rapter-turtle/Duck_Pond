import numpy as np

class load_ship_param:
    # Heron
    M = 37.758 # Mass [kg]
    I = 18.35 # Inertial tensor [kg m^2]    
    Xu = 8.9149
    Xuu = 11.2101
    Nr = 16.9542
    Nrrr = 12.8966
    Yv = 5
    Yvv = 3
    Yr = 6
    Nv = 6
    dist = 0.3   
    radius = 1
    # MPC param
    vmin = 0
    vmax = 3
    
    dt = 0.1
    N = 10

    Q = 2*np.diag([0, 1, 5, 100, 0, 25, 1e-4, 1e-4])
    R = 2*np.diag([1e-2,  # N=20
                   1e-2]) # N=20
    
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
    