from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, sqrt
from load_param import load_ship_param

def export_ship_model() -> AcadosModel:

    model_name = 'ship'

    ship_p = load_ship_param
    M = ship_p.M
    I = ship_p.I
    L = ship_p.L
    Xu = ship_p.Xu
    Xuu = ship_p.Xuu
    Nr = ship_p.Nr
    Nrr = ship_p.Nrr


    # set up states & controls
    xn   = SX.sym('xn')
    yn   = SX.sym('yn')
    psi  = SX.sym('psi')
    v    = SX.sym('v')
    r    = SX.sym('r')

    Fx  = SX.sym('Fx')
    Fn  = SX.sym('Fn')

    x = vertcat(xn, yn, psi, v, r, Fx, Fn)

    dFx  = SX.sym('dFx')
    dFn  = SX.sym('dFn')
    u   = vertcat(dFx, dFn)

    # xdot
    xn_dot  = SX.sym('xn_dot')
    yn_dot  = SX.sym('yn_dot')
    psi_dot = SX.sym('psi_dot')
    v_dot   = SX.sym('v_dot')
    r_dot   = SX.sym('r_dot')
    Fx_dot   = SX.sym('Fx_dot')
    Fn_dot   = SX.sym('Fn_dot')

    xdot = vertcat(xn_dot, yn_dot, psi_dot, v_dot, r_dot, Fx_dot, Fn_dot)

    eps = 0.00001
    # dynamics
    f_expl = vertcat(v*cos(psi),
                     v*sin(psi),
                     r,
                     ((Fx+Fn) - (Xu + Xuu*sqrt(v*v+eps))*v) / M,
                     ((-Fx+Fn)*L/2 - (Nr + Nrr*sqrt(r*r+eps))*r) / I,
                     dFx,
                     dFn
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # store meta information
    model.x_labels = ['$x$ [m]', '$y$ [m]',  '$psi$ [rad]',  '$vel.$ [m/s]', '$rot. vel.$ [rad/s]', '$Fx$ [N]', '$Fn$ [Nm]']
    model.u_labels = ['$dFx$ [N/s]', '$dFn$ [Nm/s]']
    model.t_label = '$t$ [s]'

    return model

