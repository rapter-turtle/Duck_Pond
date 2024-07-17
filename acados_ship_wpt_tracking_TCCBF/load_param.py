import numpy as np

class load_ship_param:
    # Heron
    M = 36 # Mass [kg]
    I = 8.35 # Inertial tensor [kg m^2]
    L = 0.73 # length [m]
    radius = L*0.0001
    Xu = 10
    Xuu = 16.9 # N/(m/s)^2
    Nr = 5*3
    Nrr = 13*3 # Nm/(rad/s)^2

    Fmax = 50
    Fxmax = 2*Fmax
    Fxmin = 40
    Fnmax = 20
    
    Fxmax = 60
    Fnmax = 60
    Fxmin = -60
    Fnmin = -60
    dFxmax = 5
    dFnmax = 5

    # MPC param
    dt = 0.2
    N = 10
    # Q = 2*np.diag([0, 1, 50, 250, 100, 1e-1, 1e-1])
    # R = 2*np.diag([1e2,  # N=100
    #                1e2]) # N=100

    Q = 2*np.diag([0, 1, 50, 50, 50, 1e-1, 1e-1])
    R = 2*np.diag([1e0,  # N=20
                   1e0]) # N=20
    
    # Q = 2*np.diag([0, 1, 20, 5, 50, 1e0, 1e0])
    # R = 2*np.diag([1e0,  # N=20
    #                1e0]) # N=20

    # Q = 2*np.diag([0, 1, 50, 250, 100, 1e-1, 1e-1])
    # R = 2*np.diag([1e0,  # N=10
    #                1e0]) # N=10

    # CBF param
    CBF = 2 # 0-DC 1-ED 2-TC   
    CBF_plot = 1 # 0-plot off / 1-Plot on
    #### CBF = 1 ####
    gamma1 = 1.5

    gamma2 = 0.02

    #### CBF = 2 ####
    TCCBF = 3 # 1-B1(right avoid), 2-B2, 3-B1+B2
    rmax = 0.2

    gamma_TC1 = 3
    
    # gamma_TC2 = 0.15 # N = 100
    # gamma_TC2 = 0.15 # N = 50
    # gamma_TC2 = 0.15 # N = 30
    gamma_TC2 = 0.03  # N = 20
    # gamma_TC2 = 0.025 # N = 10


    # # 임의 모델
    # M = 100 # Mass [kg]
    # I = 500 # Inertial tensor [kg m^2]
    # L = 16*0.5 # length [m]
    # radius = L/2
    # Xu = 20
    # Xuu = 50 # N/(m/s)^2
    # Nr = 25
    # Nrr = 100 # Nm/(rad/s)^2
    # Fxmax = 250
    # Fnmax = 100

    # Fxmax = 500
    # Fxmin = 100
    # Fnmax = 100

    # dFxmax = 50
    # dFnmax = 50

    # # MPC param
    # dt = 0.2
    # N = 10
    # Q = 2*np.diag([0, 1, 250, 100, 500, 1e-2, 1e-2])
    # R = 2*np.diag([1e-3, 
    #                1e-3])

    # # CBF param
    # CBF = 2 # 0-DC 1-ED 2-TC   
    # CBF_plot = 0 # 0-plot off / 1-Plot on
    # #### CBF = 1 ####
    # gamma1 = 2
    # gamma2 = 0.02

    # #### CBF = 2 ####
    # rmax = 0.3
    # gamma_TC1 = 7
    # gamma_TC2 = 0.02
    


    # #VIKNES 830 (https://core.ac.uk/download/pdf/154676362.pdf)
    # M = 3980 # Mass [kg]
    # I = 19703 # Inertial tensor [kg m^2]
    # L = 8.5 # length [m]
    # radius = L/2
    # Xu = 50
    # Xuu = 135 # N/(m/s)^2
    # Nr = 3224
    # Nrr = 3224 # Nm/(rad/s)^2
    # Fxmax = 1000
    # Fxmin = 0
    # Fnmax = 400

    # dFxmax = 100
    # dFnmax = 100

    # # MPC param
    # dt = 0.2
    # N = 10
    # Q = 2*np.diag([0, 1, 100, 50, 500, 1e-4, 1e-6])
    # R = 2*np.diag([1e-4, 
    #                1e-6])

    # # CBF param
    # CBF = 2 # 0-DC 1-ED 2-TC   
    # CBF_plot = 0 # 0-plot off / 1-Plot on
    # #### CBF = 1 ####
    # gamma1 = 2
    # gamma2 = 0.05

    # #### CBF = 2 ####
    # rmax = 0.1
    # gamma_TC1 = 5
    # gamma_TC2 = 0.025
    
    def __init__(self): 
        print(1)   
    