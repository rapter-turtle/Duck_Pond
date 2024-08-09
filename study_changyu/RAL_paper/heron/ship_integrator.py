from acados_setting import *
from load_param import *

def ship_integrator(ship, control_input, dt):

    ship_p = load_ship_param
    M = ship_p.M
    I = ship_p.I
    Xu = ship_p.Xu
    Xuu = ship_p.Xuu
    Nr = ship_p.Nr
    Nrrr = ship_p.Nrrr
    Yv = ship_p.Yv
    Yvv = ship_p.Yvv
    Yr = ship_p.Yr
    Nv = ship_p.Nv
    dist = ship_p.dist

    # set up states & controls
    psi  = ship[2]
    u    = ship[3]
    v    = ship[4]
    r    = ship[5]
    n1  = ship[6]
    n2  = ship[7]

    n1d  = control_input[0]
    n2d  = control_input[1]

    # dynamics
    xdot = np.array([u*np.cos(psi) + v*np.sin(psi),
                     u*np.sin(psi) - v*np.cos(psi),
                     r,
                     ( (n1+n2) - Xu*u - Xuu*sqrt(u*u)*u  - M*v*r)/M,
                     ( -Yv*v - Yvv*sqrt(v*v)*v - Yr*r - M*u*r)/M,
                     ( (-n1+n2)*dist - Nr*r - Nrrr*r*r*r - Nv*v)/I,
                     n1d,
                     n2d
                     ])

    ship = xdot*dt + ship

    return ship
