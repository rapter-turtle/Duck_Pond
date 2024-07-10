import numpy as np

class load_ship_param:
    # Heron
    M = 100 # Mass [kg]
    I = 500 # Inertial tensor [kg m^2]
    L = 16*0.001 # length [m]
    radius = L/2
    Xu = 20
    Xuu = 50 # N/(m/s)^2
    Nr = 25
    Nrr = 100 # Nm/(rad/s)^2
    Fxmax = 250
    Fnmax = 100

    Fxmax = 500
    Fxmin = 100
    Fnmax = 50

    dFxmax = 25
    dFnmax = 25

    # MPC param
    dt = 0.2
    N = 40
    Q = 2*np.diag([0, 1, 50, 100, 250, 0, 1e-3])
    R = 2*np.diag([1e-3, 
                   1e-3])

    # CBF param
    CBF = 2 # 0-DC 1-ED 2-TC   
    CBF_plot = 0 # 0-plot off / 1-Plot on
    #### CBF = 1 ####
    gamma1 = 1
    gamma2 = 0.02

    #### CBF = 2 ####
    rmax = 0.3
    gamma_TC1 = 5
    gamma_TC2 = 0.03
    
    def __init__(self): 
        print(1)   
    