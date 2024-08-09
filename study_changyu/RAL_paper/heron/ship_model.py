from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, sqrt
from load_param import load_ship_param

def export_ship_model() -> AcadosModel:

    model_name = 'ship'

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
    xn   = SX.sym('xn')
    yn   = SX.sym('yn')
    psi  = SX.sym('psi')
    u    = SX.sym('u')
    v    = SX.sym('v')
    r    = SX.sym('r')

    n1  = SX.sym('n1')
    n2  = SX.sym('n2')

    states = vertcat(xn, yn, psi, u, v, r, n1, n2)

    n1d  = SX.sym('n1d')
    n2d  = SX.sym('n2d')
    inputs   = vertcat(n1d, n2d)
    
    # xdot
    xn_dot  = SX.sym('xn_dot')
    yn_dot  = SX.sym('yn_dot')
    psi_dot = SX.sym('psi_dot')
    u_dot   = SX.sym('u_dot')
    v_dot   = SX.sym('v_dot')
    r_dot   = SX.sym('r_dot')
    n1_dot   = SX.sym('n1_dot')
    n2_dot   = SX.sym('n2_dot')

    states_dot = vertcat(xn_dot, yn_dot, psi_dot, u_dot, v_dot, r_dot, n1_dot, n2_dot)

    eps = 0.00001
    # dynamics
    f_expl = vertcat(u*cos(psi) + v*sin(psi),
                     u*sin(psi) - v*cos(psi),
                     r,
                     ( (n1+n2) - Xu*u - Xuu*sqrt(u*u+eps)*u - M*v*r )/M,
                     ( -Yv*v - Yvv*sqrt(v*v+eps)*v - Yr*r  - M*u*r)/M,
                     ( (-n1+n2)*dist - Nr*r - Nrrr*r*r*r - Nv*v)/I,
                     n1d,
                     n2d
                     )

    f_impl = states_dot - f_expl
    
    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = inputs
    model.name = model_name

    # store meta information
    model.x_labels = ['$x$ [m]', '$y$ [m]',  '$psi$ [rad]',  '$u$ [m/s]', '$v$ [m/s]', '$r$ [rad/s]', '$n_1$ [N]', '$n_2$ [N]']
    model.u_labels = ['$n_1_d$ [N/s]', '$n_2_d$ [N/s]']
    model.t_label = '$t$ [s]'

    return model

