from plot_asv import *
from gen_ref import *
from acados_setting import *

def recover_simulator(ship, tship, control_input, dt, t, extra_control, disturbance_state):

    M = 36 # Mass [kg]
    I = 8.35 # Inertial tensor [kg m^2]
    L = 0.73 # length [m]
    Xu = 10
    Xuu = 16.9 # N/(m/s)^2
    Nr = 5
    Nrr = 13 # Nm/(rad/s)^2


    # set up states & controls
    psi  = ship[2]
    v    = ship[3]
    r    = ship[4]
    n1  = ship[5]
    n2  = ship[6]


    ################## Disturbance ##################
    wind_direction = 60*3.141592/180
    wind_speed = 5.0

    ## Wave Disturbance ## 
    # wave_disturbance(disturbance_state, wave_direction, wind_speed, omega, lamda, Kw, sigmaF1, sigmaF2, dt):
    disturbance_state[:3], XY_wave_force = wave_disturbance(disturbance_state[:3], wind_direction, 7.0, 0.8, 0.1, 2.0, 10, 2, dt)
    disturbance_state[3:6], N_wave_force = wave_disturbance(disturbance_state[3:6], wind_direction, 7.0, 0.8, 0.1, 1.0, 1, 0.1, dt)
    
    X_wave_force = XY_wave_force*np.cos(wind_direction - psi)
    Y_wave_force = XY_wave_force*np.sin(wind_direction - psi)

    U_wave_force = X_wave_force*np.sin(psi) + Y_wave_force*np.cos(psi)
    



    ## Wind Disturbance ## 
    Afw = 0.3  #Frontal area
    Alw = 0.44  #Lateral area
    LOA = 1.3  #Length
    lau = 1.2  # air density
    CD_lAF = 0.55
    delta = 0.6
    CDl = CD_lAF*Afw/Alw
    CDt = 0.9

    u_rel_wind = v - wind_speed*np.cos(wind_direction - psi)
    v_rel_wind = - wind_speed*np.sin(wind_direction - psi)
    gamma = -np.arctan2(v_rel_wind, u_rel_wind)

    Cx = CD_lAF*np.cos(gamma)/(1 - delta*0.5*(1-CDl/CDt)*(np.sin(2*gamma))**2)
    Cy = CDt*np.sin(gamma)/(1 - delta*0.5*(1-CDl/CDt)*(np.sin(2*gamma))**2)
    Cn = -0.18*(gamma - np.pi*0.5)*Cy

    X_wind_force = 0.5*lau*(u_rel_wind**2 + v_rel_wind**2)*Cx*Afw
    Y_wind_force = 0.5*lau*(u_rel_wind**2 + v_rel_wind**2)*Cy*Alw
    N_wind_force = 0.5*lau*(u_rel_wind**2 + v_rel_wind**2)*Cn*Alw*LOA    

    U_wind_force = X_wind_force + X_wind_force*np.sin(psi) + Y_wind_force*np.cos(psi)


    ######################################################
    
    # U_wave_force = 0.0 
    # N_wave_force = 0.0

    # U_wind_force = 0.0   
    # N_wind_force = 0.0
    
    disturbance = np.array([U_wave_force + U_wind_force, N_wave_force + N_wind_force])
    

    n1d  = control_input[0]
    n2d  = control_input[1]

    n1_extra = extra_control[0]
    n2_extra = extra_control[1]


    # dynamics
    xdot = np.array([v*np.cos(psi),
                     v*np.sin(psi),
                     r,
                     ((n1+n2)-(Xu + Xuu*np.sqrt(v*v))*v + disturbance[0] + (n1_extra + n2_extra))/M ,
                     ((-n1+n2)*L/2 - (Nr + Nrr*np.sqrt(r*r))*r + disturbance[1] + (-n1_extra + n2_extra)*L/2)/I,
                     n1d,
                     n2d
                     ])

    ship = xdot*dt + ship


    tpsi  = tship[2]
    tu    = tship[3]

    tship_xdot = np.array([tu*np.cos(tpsi),
                     tu*np.sin(tpsi),
                     0.0,
                     0.0
                     ])


    tship = tship_xdot*dt + tship

    return ship, tship, disturbance_state





def wave_disturbance(disturbance_state, wave_direction, wind_speed, omega, lamda, Kw, sigmaF1, sigmaF2, dt):
    
    omega_e = np.abs(omega - (omega*omega/9.81)*wind_speed*np.cos(wave_direction))
    
    x1 = disturbance_state[0]
    x2 = disturbance_state[1]

    omegaF1 = np.random.normal(0.0, sigmaF1)
    omegaF2 = np.random.normal(0.0, sigmaF2)

    xdot = np.array([x2, -omega_e*omega_e*x1 - 2*lamda*omega_e*x2 + Kw*omegaF1, omegaF2 ])
    disturbance_state = xdot*dt + disturbance_state 


    disturbance_force = disturbance_state[0] + disturbance_state[2]

    return disturbance_state, disturbance_force