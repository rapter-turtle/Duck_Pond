from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, sqrt
from load_param import load_kinematic_param

def export_kinematic_model() -> AcadosModel:

    model_name = 'kinematic'

    # set up states & controls
    xn   = SX.sym('xn')
    yn   = SX.sym('yn')
    psi  = SX.sym('psi')
    u    = SX.sym('u')
    rot  = SX.sym('w')
    acc  = SX.sym('a')

    x = vertcat(xn, yn, psi, u, rot, acc)

    drot  = SX.sym('drot')
    dacc  = SX.sym('dacc')
    con  = vertcat(drot, dacc)

    # xdot
    xn_dot  = SX.sym('xn_dot')
    yn_dot  = SX.sym('yn_dot')
    psi_dot = SX.sym('psi_dot')
    u_dot   = SX.sym('u_dot')
    rot_dot   = SX.sym('rot_dot')
    acc_dot   = SX.sym('acc_dot')

    xdot = vertcat(xn_dot, yn_dot, psi_dot, u_dot, rot_dot, acc_dot)

    # dynamics
    f_expl = vertcat(u*cos(psi),
                     u*sin(psi),
                     rot,
                     acc,
                     drot,
                     dacc
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = con
    model.name = model_name

    return model

