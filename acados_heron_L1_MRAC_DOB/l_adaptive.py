import numpy as np
import math 

def L1_control(state, state_estim, param_filtered, dt, param_estim):

 
    M = 36 # Mass [kg]
    I = 8.35 # Inertial tensor [kg m^2]
    L = 0.73 # length [m]
    Xu = 10
    Xuu = 16.9 # N/(m/s)^2
    Nr = 5
    Nrr = 13 # Nm/(rad/s)^2

    w_cutoff = 2
    u_max = 1.2
    r_max = 0.8

    # set up states & controls
    xn   = state_estim[0]
    yn   = state_estim[1]
    psi  = state_estim[2]
    v    = state_estim[3]
    r    = state_estim[4]
    n1  = state[5]
    n2  = state[6]


    eps = 0.00001
    # dynamics
    xdot = np.array([v*np.cos(psi),
                     v*np.sin(psi),
                     r,
                     ((n1+n2) - (Xu + Xuu*np.sqrt(v*v+eps))*v)/M + param_estim[0] + param_filtered[0],
                     (((-n1+n2))*L/2 - (Nr + Nrr*np.sqrt(r*r+eps))*r)/I + param_estim[1] + param_filtered[1] 
                     ])

    
    Am = -np.eye(5)
    state_error =  state_estim - state[:len(state_estim)]
    # print(state_error) 

    x_plus = (xdot + np.dot(Am,state_error))*dt + state_estim

    before_param_estim = param_estim

    gain = -0.0000001
    pi = (1/gain)*(np.exp(gain*dt)-1)
    param_estim[0] = -np.exp(gain*dt)*state_error[3]/pi
    param_estim[1] = -np.exp(gain*dt)*state_error[4]/pi

    before_param_filtered = param_filtered
    param_filtered = before_param_filtered*math.exp(-w_cutoff*dt) - param_estim*(1-math.exp(-w_cutoff*dt))

    return x_plus, param_estim, param_filtered


def h_function(x, eta, x_max):
    h = ((eta + 1)*x*x - (x_max*x_max))/(eta*x_max*x_max)
    hdot = 2*(eta + 1)*x/(eta*x_max*x_max)
    return h, hdot


def param_dynamics(x_error, param_estim, g, input_max):
    
    P = 0.5*np.eye(5)
    L = 100
    param_update = -np.dot(np.dot(g, P), x_error)
    # print(param_update)
    
    h, hdot = h_function(param_estim, 0.001, input_max)
    # param_dot = 0.0
    if h >= 0 and param_update*hdot > 0:
        param_dot = 0#(param_update - (hdot/np.abs(hdot))*np.abs(param_update*h))
        print("aa : ", param_dot)
        print("hdot : ", hdot)

    # elif h >= 0 and param_update*hdot <= 0:
    #     param_dot = param_update
    else:
        param_dot = param_update
    param_dot = param_update
    
    return param_dot


