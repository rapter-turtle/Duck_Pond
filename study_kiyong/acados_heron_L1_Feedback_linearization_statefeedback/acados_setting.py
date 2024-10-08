from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from heron_model import export_heron_model
import scipy.linalg
import numpy as np
from casadi import SX, vertcat, cos, sin


def setup_recovery(x0, Fmax, N_horizon, Tf):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_heron_model()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = N_horizon

    # set cost module
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    Q_mat = 2*np.diag([1, 1, 5, 5])
    R_mat = 2*np.diag([1e-1, 1e-1])

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat

    ocp.model.cost_y_expr = vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref  = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e, ))

    ocp.constraints.x0 = x0

    # set up parameters
    ocp.parameter_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 
    param = model.p

    #Normal boundary constraints
    h_expr = (model.x[0]*param[0]) + (model.x[1]*param[1]) + param[2]

    ocp.model.con_h_expr = h_expr
 
    # For CBF
    ocp.constraints.uh = 1e10*np.ones(1)
    ocp.constraints.lh = -1e10*np.ones(1)

    ocp.constraints.idxsh = np.array([0])
    ocp.constraints.idxsh_e = np.array([0])
    Zh = 1e2*np.ones(1)#1e1*np.ones(1)
    zh = 1e2*np.ones(1)#1e1*np.ones(1)
    ocp.cost.zl = zh
    ocp.cost.zu = zh
    ocp.cost.Zl = Zh
    ocp.cost.Zu = Zh
    ocp.cost.zl_e = zh
    ocp.cost.zu_e = zh
    ocp.cost.Zl_e = Zh
    ocp.cost.Zu_e = Zh

    # copy for terminal
    ocp.constraints.uh_e = ocp.constraints.uh
    ocp.constraints.lh_e = ocp.constraints.lh
    ocp.model.con_h_expr_e = ocp.model.con_h_expr

    # set constraints
    ocp.constraints.lbu = np.array([-Fmax,-Fmax])
    ocp.constraints.ubu = np.array([+Fmax,+Fmax])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([-5, -5])
    ocp.constraints.ubx = np.array([5, 5])
    ocp.constraints.idxbx = np.array([2, 3])

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_newton_iter = 50
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf

    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return acados_ocp_solver, acados_integrator
