from acados_setting import *
from load_param import *

def ship_integrator(ship, control_input, dt):

    ship_p = load_ship_param
    M = ship_p.M
    I = ship_p.I
    L = ship_p.L
    Xu = ship_p.Xu
    Xuu = ship_p.Xuu
    Nr = ship_p.Nr
    Nrr = ship_p.Nrr


    # set up states & controls
    psi  = ship[2]
    v    = ship[3]
    r    = ship[4]
    Fx  = ship[5]
    Fn  = ship[6]

    dFx  = control_input[0]
    dFn  = control_input[1]


    # dynamics
    xdot = np.array([v*np.cos(psi),
                     v*np.sin(psi),
                     r,
                     ((Fx+Fn)-(Xu + Xuu*np.sqrt(v*v))*v)/M ,
                     ((-Fx+Fn)*L/2 - (Nr + Nrr*np.sqrt(r*r))*r)/I,
                     dFx,
                     dFn
                     ])

    ship = xdot*dt + ship

    return ship
