import numpy as np
import math 

def estim_update_state(x_t, x_error,  u, l1_u, dt, param_estim,t):


    m = 3980
    Iz = 19703
    xdot = 0
    Yvdot = 0
    Yrdot = 0
    Nvdot = 0
    Nrdot = 0
    Xu = -50
    Yv = -200
    Yr = 0
    Nv = 0
    Nr = -1281
    m11 = m - xdot
    m22 = m - Yvdot 
    m23 = -Yrdot
    m32 = -Nvdot
    m33 = Iz - Nrdot

    l = 3.5

    et1 = x_t[0]
    et2 = x_t[1]
    et3 = x_t[2]
    et4 = x_t[3]
    psi = x_t[4]
    r = x_t[5]
    Tau_x = u[0]
    Tau_y = u[1] 

    uu = et3*math.cos(psi) + et4*math.sin(psi) + math.cos(psi)
    v = -et3*math.sin(psi) + et4*math.cos(psi) - r*l - math.sin(psi)

    M = np.array([[m11, 0, 0], [0, m22, m23], [0, m32, m33]])
    
    Cv = np.array([[0, 0, -m22*v-m23*r], [0, 0, m11*uu], [m22*v+m23*r, -m11*uu, 0]])
    
    D = -np.array([[Xu, 0, 0], [0, Yv, Yr], [0, Nv, Nr]])
    
    R = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]])

    uvr = np.array([uu, v, r])

    M_inv = np.linalg.inv(M)


    Tau = np.array([Tau_x, Tau_y, -4*(Tau_y)])

    r_uvr_dot_bM = (Tau - np.dot(Cv, uvr) - np.dot(D, uvr))
    r_uvr_dot = np.dot(M_inv, r_uvr_dot_bM)

    uvr_dot_bM = (- np.dot(Cv, uvr) - np.dot(D, uvr))
    uvr_dot = np.dot(M_inv, uvr_dot_bM)

    et1dot = uu*math.cos(psi) - v*math.sin(psi) - r*l*math.sin(psi) - 1

    et2dot = uu*math.sin(psi) + v*math.cos(psi) + r*l*math.cos(psi)

    et3dot = uvr_dot[0]*math.cos(psi) - uvr_dot[1]*math.sin(psi) - uvr_dot[2]*l*math.sin(psi) - uu*r*math.sin(psi) - v*r*math.cos(psi) - r*r*l*math.cos(psi) + Tau_x*math.cos(psi)/m - (1/m - 4*l/Iz)*Tau_y*math.sin(psi) + param_estim[0] + l1_u[0]
    
    et4dot = uvr_dot[0]*math.sin(psi) + uvr_dot[1]*math.cos(psi) + uvr_dot[2]*l*math.cos(psi) + uu*r*math.cos(psi) - v*r*math.sin(psi) - r*r*l*math.sin(psi) + Tau_x*math.sin(psi)/m + (1/m - 4*l/Iz)*Tau_y*math.cos(psi) + param_estim[1] + l1_u[1]
    

    Am = -np.eye(6)

    xdot = np.concatenate(([et1dot],[et2dot],[et3dot], [et4dot],[r],[r_uvr_dot[2] + param_estim[2]]), axis=0) + np.dot(Am, x_error)
    
    
    x_t_plus_1 = xdot*dt + x_t

    real_psi = x_t_plus_1[4]
    if real_psi > math.pi:
        x_t_plus_1[4] -= 2*math.pi
    if real_psi < -math.pi:
        x_t_plus_1[4] += 2*math.pi
        


    return x_t_plus_1


def h_function(x, eta, x_max):
    h = x*x - (x_max*x_max)

    # hdot = (2*x)/(eta*x_max*x_max)
    hdot = 2*x

    return h, hdot


def param_dynamics(x_error, param_estim, g, input_max):
    
    P = 0.5*np.eye(6)

    param_update = -np.dot(np.dot(g, P), x_error)
    
    h, hdot = h_function(param_estim, 0.1, input_max)
    # if g[2] == 1:
    #     print("p : ", param_update)
    #     print("h : ", h)

    # print(h)
    if h > 0 and param_update*hdot > 0:
        param_dot = 0

    else:
        param_dot = param_update


    return param_dot


