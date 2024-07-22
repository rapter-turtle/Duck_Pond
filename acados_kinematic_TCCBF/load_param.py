import numpy as np

class load_kinematic_param:
    radius = 0.001
    accmax = 0.3
    accmin = -0.3
    daccmax = 1
    drotmax = 1
    vmin = 1
    vmax = 3
    # MPC param
    dt = 0.2
    N = 10

    Q = 2*np.diag([0, 1, 10, 100, 5e1, 5e1])
    R = 2*np.diag([5e1,  # N=20
                   5e1]) # N=20
    
    # CBF param
    CBF = 2 # 0-DC 1-ED 2-TC   
    CBF_plot = 1 # 0-plot off / 1-Plot on

    #### CBF = 1 ####
    gamma1 = 1
    gamma2 = 0.15

    #### CBF = 2 ####
    TCCBF = 3 # 1-B1(right avoid), 2-B2, 3-B1+B2
    rmax = 0.2
    gamma_TC1 = 1
    gamma_TC2 = 0.15
    
    def __init__(self): 
        print(1)   
    