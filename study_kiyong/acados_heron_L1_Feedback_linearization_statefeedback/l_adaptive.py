import numpy as np
import math 

def L1_control(state, state_estim, param_filtered, dt, param_estim, MPC_control):

 
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
    head_dist = 1.0

    w_cutoff = 3


    # set up states & controls
    xn  = state[0]
    yn  = state[1]
    psi  = state[2]
    u    = state[3]
    v    = state[4]
    r    = state[5]
    n1  = MPC_control[0]
    n2  = MPC_control[1]


    f_usv = np.array([(-(Xu + Xuu*np.sqrt(u*u))*u)/M,
          ( -Yv*v - Yvv*np.sqrt(v*v)*v - Yr*r)/M,
          (- (Nr + Nrrr*r*r)*r - Nv*v)/I
          ])

    virtual_state = np.array([xn + head_dist*np.cos(psi),
                              yn + head_dist*np.sin(psi),
                              u*np.cos(psi) - v*np.sin(psi) - head_dist*r*np.sin(psi),
                              u*np.sin(psi) + v*np.cos(psi) + head_dist*r*np.cos(psi)])


    virtual_control = np.array([0.0,
                                0.0,
                                f_usv[0]*np.cos(psi) - u*r*np.sin(psi) - f_usv[1]*np.sin(psi) - v*r*np.cos(psi) - head_dist*f_usv[2]*np.sin(psi) - head_dist*r*r*np.cos(psi),
                                f_usv[0]*np.sin(psi) + u*r*np.cos(psi) + f_usv[1]*np.cos(psi) - v*r*np.sin(psi) + head_dist*f_usv[2]*np.cos(psi) - head_dist*r*r*np.sin(psi)])

    nominal_control = np.array([0.0,
                                0.0,
                                ((n1+n2)/M)*np.cos(psi)- head_dist*((-n1+n2)*dist/I)*np.sin(psi),
                                ((n1+n2)/M)*np.sin(psi)+ head_dist*((-n1+n2)*dist/I)*np.cos(psi)])


    x_error = state_estim - virtual_state 
    
    state_feedback = np.array([0.0,
                               0.0,
                               -virtual_state[0]-1.73*virtual_state[2],
                               -virtual_state[1]-1.73*virtual_state[3]])
    
    adaptive_control = param_filtered[2:4] - virtual_control[2:4] + nominal_control[2:4] #+ state_feedback[2:4]

    L1_thruster = np.array([0.5*(M*np.cos(psi) + I*np.sin(psi)/(head_dist*dist))*adaptive_control[0] + 0.5*(M*np.sin(psi) - I*np.cos(psi)/(head_dist*dist))*adaptive_control[1],
                            0.5*(M*np.cos(psi) - I*np.sin(psi)/(head_dist*dist))*adaptive_control[0] + 0.5*(M*np.sin(psi) + I*np.cos(psi)/(head_dist*dist))*adaptive_control[1]
                            ])    
    
    A = np.array([state_estim[2], state_estim[3],0,0])
    xdot = A + param_estim + param_filtered - x_error + nominal_control #+ state_feedback 

    x_t_plus = xdot*dt + state_estim
    

    gain = -1.0
    pi = (1/gain)*(np.exp(gain*dt)-1.0)
    param_estim = -np.exp(gain*dt)*x_error/pi
    
    before_param_filtered = param_filtered
    param_filtered = before_param_filtered*math.exp(-w_cutoff*dt) - param_estim*(1-math.exp(-w_cutoff*dt))


    
    return x_t_plus, param_estim, param_filtered, L1_thruster





