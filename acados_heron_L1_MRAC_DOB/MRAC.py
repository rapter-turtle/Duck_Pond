import numpy as np
import math 

def MRAC_control(state, state_estim, dt, param_estim):


    M = 36 # Mass [kg]
    I = 8.35 # Inertial tensor [kg m^2]
    L = 0.73 # length [m]
    Xu = 10
    Xuu = 16.9 # N/(m/s)^2
    Nr = 5
    Nrr = 13 # Nm/(rad/s)^2

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
                     ((n1+n2) - (Xu + Xuu*np.sqrt(v*v+eps))*v)/M,
                     (((-n1+n2))*L/2 - (Nr + Nrr*np.sqrt(r*r+eps))*r)/I 
                     ])

    Am = -np.eye(5)
    state_error =  state_estim - state[:len(state_estim)]
    # print(state_error) 

    x_plus = (xdot + np.dot(Am,state_error))*dt + state_estim

    before_param_estim = param_estim
    pu = 400*dt*param_dynamics(state_error, before_param_estim[0], np.array([0, 0, 0, 1, 0]), 1.0) + before_param_estim[0]
    pr = 400*dt*param_dynamics(state_error, before_param_estim[1], np.array([0, 0, 0, 0, 1]), 1.0) + before_param_estim[1]

    param_estim[0] = pu
    param_estim[1] = pr

    return x_plus, param_estim


def h_function(x, eta, x_max):
    h = x*x - (x_max*x_max)
    hdot = 2*x
    return h, hdot


def param_dynamics(x_error, param_estim, g, input_max):
    
    P = 0.5*np.eye(5)

    param_update = -np.dot(np.dot(g, P), x_error)
    
    h, hdot = h_function(param_estim, 0.1, input_max)
    # if h > 0 and param_update*hdot > 0:
    #     param_dot = 0

    # else:
    #     param_dot = param_update
    # print(param_dot)
    param_dot = param_update
    return param_dot


