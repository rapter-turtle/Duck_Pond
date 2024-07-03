import numpy as np
import math 

def update_state(x_t, u, l1_u, dt, V, V_t,t):


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


    uu = x_t[0]
    v = x_t[1]
    r = x_t[2]
    psi = x_t[5]
    Tau_x = x_t[6]
    Tau_y = x_t[7]  



    M = np.array([[m11, 0, 0], [0, m22, m23], [0, m32, m33]])
    
    Cv = np.array([[0, 0, -m22*v-m23*r], [0, 0, m11*uu], [m22*v+m23*r, -m11*uu, 0]])
    
    D = -np.array([[Xu, 0, 0], [0, Yv, Yr], [0, Nv, Nr]])
    
    R = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]])

    uvr = np.array([uu, v, r])



    # Disturbance
    direction = -60*math.pi/180
    lau = 1.2
    wind_v = 5.0
    Al = 9.0
    Af = 3.0
    uw = wind_v*math.cos(direction - psi)
    vw = wind_v*math.sin(direction - psi)
    urw = uu - uw
    vrw = v - vw
    wind_rv = math.sqrt(urw**2+vrw**2)
    gamma = -math.atan2(vrw,urw)

    windx = -Af*wind_rv**2*0.7*math.cos(gamma) 
    windy = Al*wind_rv**2*0.8*math.sin(gamma)
    windpsi = Af*wind_rv**2*0.2*math.sin(2*gamma)

    wave_force = 50.0*math.cos(0.5*t) + 20.0*math.cos(1*t) + 70.0*math.cos(0.2*t) 
    wave_direction =45*math.pi/180
    paramx = wave_force*math.sin(wave_direction) + windx
    paramy = wave_force*math.cos(wave_direction) + windy
    parampsi =  windpsi

    disturbance_x = (paramy*math.sin(psi) + paramx*math.cos(psi))
    disturbance_y = (paramy*math.cos(psi) - paramx*math.sin(psi))
    disturbance_psi = parampsi

  
    dis = np.array([disturbance_x, disturbance_y, disturbance_psi])




    # Control input
    Tau = np.array([Tau_x  + l1_u[0], Tau_y + l1_u[1], -4*(Tau_y+ l1_u[1])])
    Tau = Tau + np.array([u[0], u[1], -4*u[1]])*dt

    if Tau[0] > 799.999:
        Tau[0] = 799.99
    if Tau[0] < -799.999:
        Tau[0] = -799.99

    if Tau[1] > 199.999:
        Tau[1] = 199.99
    if Tau[1] < -199.999:
        Tau[1] = -199.99

    M_inv = np.linalg.inv(M)

    uvr_dot_bM = (dis + Tau - np.dot(Cv, uvr) - np.dot(D, uvr))
    uvr_dot = np.dot(M_inv, uvr_dot_bM)
    xypsi_dot = np.dot(R, uvr) 

    V_y = V_t[1]

    V_t[1] = V_y

    xdot = np.concatenate((uvr_dot, xypsi_dot, u, V_t), axis=0)
    

    x_t_plus_1 = xdot * dt + x_t

    real_psi = x_t_plus_1[5]
    if real_psi > math.pi:
        x_t_plus_1[5] -= 2*math.pi
    if real_psi < -math.pi:
        x_t_plus_1[5] += 2*math.pi
        
    
    if x_t_plus_1[6] > 799.999:
        x_t_plus_1[6] = 799.99
    if x_t_plus_1[6] < -799.999:
        x_t_plus_1[6] = -799.99

    if x_t_plus_1[7] > 199.999:
        x_t_plus_1[7] = 199.99
    if x_t_plus_1[7] < -199.999:
        x_t_plus_1[7] = -199.99




    V_ship = np.array([1.0, V_y, 0.0])

    return x_t_plus_1, V_ship, Tau, dis

