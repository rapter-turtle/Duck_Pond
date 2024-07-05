from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, sqrt

def export_heron_model() -> AcadosModel:

    model_name = 'heron'

    # constants
    M = 36 # Mass [kg]
    I = 8.35 # Inertial tensor [kg m^2]
    L = 0.73 # length [m]
    Xu = 10
    Xuu = 16.9 # N/(m/s)^2
    Nr = 5
    Nrr = 13 # Nm/(rad/s)^2


    # set up states & controls
    xn   = SX.sym('xn')
    yn   = SX.sym('yn')
    psi  = SX.sym('psi')
    v    = SX.sym('v')
    r    = SX.sym('r')

    n1  = SX.sym('n1')
    n2  = SX.sym('n2')

    x = vertcat(xn, yn, psi, v, r, n1, n2)

    n1d  = SX.sym('n1d')
    n2d  = SX.sym('n2d')
    u   = vertcat(n1d, n2d)

    # xdot
    xn_dot  = SX.sym('xn_dot')
    yn_dot  = SX.sym('yn_dot')
    psi_dot = SX.sym('psi_dot')
    u_dot   = SX.sym('u_dot')
    r_dot   = SX.sym('r_dot')
    n1_dot   = SX.sym('n1_dot')
    n2_dot   = SX.sym('n2_dot')

    xdot = vertcat(xn_dot, yn_dot, psi_dot, u_dot, r_dot, n1_dot, n2_dot)

    eps = 0.00001
    # dynamics
    f_expl = vertcat(v*cos(psi),
                     v*sin(psi),
                     r,
                     ((n1+n2)-(Xu + Xuu*sqrt(v*v+eps))*v)/M,
                     ((-n1+n2)*L/2 - (Nr + Nrr*sqrt(r*r+eps))*r)/I,
                     n1d,
                     n2d
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
    model.x_labels = ['$x$ [m]', '$y$ [m]',  '$psi$ [rad]',  '$u$ [m/s]', '$r$ [rad/s]', '$n_1$ [N]', '$n_2$ [N]']
    model.u_labels = ['$n_1_d$ [N/s]', '$n_2_d$ [N/s]']
    model.t_label = '$t$ [s]'

    return model

