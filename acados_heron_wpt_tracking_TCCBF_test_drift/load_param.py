import numpy as np

class load_ship_param:
    # Heron
    M = 36 # Mass [kg]
    I = 8.35 # Inertial tensor [kg m^2]
    L = 0.73 # length [m]
    radius = L*0.0001
    Xu = 10
    Xuu = 16.9 # N/(m/s)^2
    Yv = 20
    Nr = 15
    Nrr = 40 # Nm/(rad/s)^2
    
    Fxmax = 60
    Fnmax = 60
    Fxmin = -60
    Fnmin = -60
    dFxmax = 5
    dFnmax = 5

    # MPC param
    dt = 0.2
    N = 10

    Q = 2*np.diag([0, 2, 5, 100, 0, 10, 1e-4, 1e-4])
    R = 2*np.diag([1e-3,  # N=20
                   1e-3]) # N=20
    
    # CBF param
    CBF = 2 # 0-DC 1-ED 2-TC   
    CBF_plot = 1 # 0-plot off / 1-Plot on

    #### CBF = 1 ####
    gamma1 = 1.5
    gamma2 = 0.05

    #### CBF = 2 ####
    TCCBF = 3 # 1-B1(right avoid), 2-B2, 3-B1+B2
    rmax = 0.25
    gamma_TC1 = 2
    gamma_TC2 = 0.07

    
    def __init__(self): 
        print(1)   
    