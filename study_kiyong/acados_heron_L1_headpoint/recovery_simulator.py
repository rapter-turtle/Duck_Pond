from plot_asv import *
from gen_ref import *
from acados_setting import *

def recover_simulator(ship, tship, control_input, dt, disturbance, extra_control):

    # constants
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


    # set up states & controls
    psi  = ship[2]
    u    = ship[3]
    v    = ship[4]
    r    = ship[5]
    n1  = ship[6]
    n2  = ship[7]

    n1d  = control_input[0]
    n2d  = control_input[1]

    n1_extra = 0.0#extra_control[0]
    n2_extra = 0.0#extra_control[1]

    # dynamics
    xdot = np.array([u*cos(psi) - v*sin(psi),
                     u*sin(psi) + v*cos(psi),
                     r,
                     ((n1+n2)-(Xu + Xuu*np.sqrt(u*u))*u + disturbance[0] + (n1_extra + n2_extra))/M ,
                     ( -Yv*v - Yvv*np.sqrt(v*v)*v - Yr*r + disturbance[1])/M,
                     ((-n1+n2)*dist - (Nr + Nrrr*r*r)*r - Nv*v + disturbance[2] + (-n1_extra + n2_extra)*dist)/I,
                     n1d,
                     n2d
                     ])


    ship = xdot*dt + ship
    ppsi = ship[2]
    
    if ppsi > 3.141592:
        ship[2] = ppsi - 2*3.141592
    if ppsi < -3.141592:
        ship[2] = ppsi + 2*3.141592
    

    tpsi  = tship[2]
    tu    = tship[3]

    tship_xdot = np.array([tu*np.cos(tpsi),
                     tu*np.sin(tpsi),
                     0.0,
                     0.0
                     ])


    tship = tship_xdot*dt + tship

    return ship, tship



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