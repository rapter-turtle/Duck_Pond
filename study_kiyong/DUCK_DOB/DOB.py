import numpy as np
import math 

def DOB(state, state_estim, dt, param_estim, MPC_control, param_filtered):

 
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


    # set up states & controls
    xn  = state[0]
    yn  = state[1]
    psi  = -state[2]
    u    = state[3]
    v    = state[4]
    r    = state[5]

    n1  = 0.0#MPC_control[0]
    n2  = 0.0#MPC_control[1]


    f_usv = np.array([(-(Xu + Xuu*np.sqrt(u*u))*u)/M,
          ( -Yv*v - Yvv*np.sqrt(v*v)*v - Yr*r)/M,
          (- (Nr + Nrrr*r*r)*r - Nv*v)/I
          ])

    virtual_state = np.array([0.0,0.0,
                              u*np.cos(psi) - v*np.sin(psi) - head_dist*r*np.sin(psi),
                              u*np.sin(psi) + v*np.cos(psi) + head_dist*r*np.cos(psi)])

    virtual_control = np.array([0.0,0.0,
                                f_usv[0]*np.cos(psi) - u*r*np.sin(psi) - f_usv[1]*np.sin(psi) - v*r*np.cos(psi) - head_dist*f_usv[2]*np.sin(psi) - head_dist*r*r*np.cos(psi),
                                f_usv[0]*np.sin(psi) + u*r*np.cos(psi) + f_usv[1]*np.cos(psi) - v*r*np.sin(psi) + head_dist*f_usv[2]*np.cos(psi) - head_dist*r*r*np.sin(psi)])

    nominal_control = np.array([0.0,0.0,
                                ((n1+n2)/M)*np.cos(psi) - head_dist*((-n1+n2)*dist/I)*np.sin(psi),
                                ((n1+n2)/M)*np.sin(psi) + head_dist*((-n1+n2)*dist/I)*np.cos(psi)])
    

    x_error = state_estim - virtual_state 

    

    xdot = param_estim + virtual_control

    x_t_plus = xdot*dt + state_estim
    
    w_cutoff = 0.5
    before_param_filtered = param_filtered
    param_filtered = before_param_filtered*math.exp(-w_cutoff*dt) - param_estim*(1-math.exp(-w_cutoff*dt))

#     gain = -1.0

    gain = -1
    pi = (1/gain)*(np.exp(gain*dt)-1.0)
    param_estim = -np.exp(gain*dt)*x_error/pi
    

    return x_t_plus, param_estim, param_filtered




