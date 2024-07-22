from acados_setting import *
from load_param import *

def ship_integrator(ship, control_input, dt):

    ship_p = load_ship_param
    M = ship_p.M
    I = ship_p.I
    L = ship_p.L
    Xu = ship_p.Xu
    Xuu = ship_p.Xuu
    Yv = ship_p.Yv
    Nr = ship_p.Nr
    Nrr = ship_p.Nrr


    # set up states & controls
    psi  = ship[2]
    u    = ship[3]
    v    = ship[4]
    r    = ship[5]
    Fx  = ship[6]
    Fn  = ship[7]

    dFx  = control_input[0]
    dFn  = control_input[1]


    # dynamics
    xdot = np.array([u*np.cos(psi) + v*np.sin(psi),
                     u*np.sin(psi) - v*np.cos(psi),
                     r,
                     ((Fx)+M*v*r-(Xu + Xuu*np.sqrt(u*u))*u)/M ,
                     (-M*u*r - Yv*v) / M,
                     ((Fn)*L/2 - (Nr + Nrr*np.sqrt(r*r))*r)/I,
                     dFx,
                     dFn
                     ])

    ship = xdot*dt + ship

    return ship
