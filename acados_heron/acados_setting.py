from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from heron_model import export_heron_model
import scipy.linalg
import numpy as np
from casadi import vertcat

def setup(x0, Fmax, N_horizon, Tf):
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

    Q_mat = 2*np.diag([2, 2, 50, 10, 5, 1e-4, 1e-4])
    R_mat = 2*np.diag([1e-6, 1e-6])

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat

    ocp.model.cost_y_expr = vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref  = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e, ))

    ocp.constraints.x0 = x0


    obs_rad = 0.5
    obs_x = 0.0
    obs_y = 0.0
    circle = (obs_x, obs_y, obs_rad)
    ocp.constraints.uh = np.array([1e10]) # doenst matter
    ocp.constraints.lh = np.array([0])
    x_square = (model.x[0]-obs_x) ** 2 + (model.x[1] - obs_y) ** 2 - obs_rad**2
    ocp.model.con_h_expr = x_square

    # copy for terminal
    ocp.constraints.uh_e = ocp.constraints.uh
    ocp.constraints.lh_e = ocp.constraints.lh
    ocp.model.con_h_expr_e = ocp.model.con_h_expr

    # soften
    ocp.constraints.idxsh = np.array([0])
    ocp.constraints.idxsh_e = np.array([0])
    Zh = 1e6 * np.ones(1)
    zh = 1e4 * np.ones(1)
    ocp.cost.zl = zh
    ocp.cost.zu = zh
    ocp.cost.Zl = Zh
    ocp.cost.Zu = Zh
    ocp.cost.zl_e = np.concatenate((ocp.cost.zl_e, zh))
    ocp.cost.zu_e = np.concatenate((ocp.cost.zu_e, zh))
    ocp.cost.Zl_e = np.concatenate((ocp.cost.Zl_e, Zh))
    ocp.cost.Zu_e = np.concatenate((ocp.cost.Zu_e, Zh))




    # set constraints
    ocp.constraints.lbu = np.array([-Fmax/5,-Fmax/5])
    ocp.constraints.ubu = np.array([+Fmax/5,+Fmax/5])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([-1, -0.5, -Fmax, -Fmax])
    ocp.constraints.ubx = np.array([2, 0.5, Fmax, Fmax])
    ocp.constraints.idxbx = np.array([3, 4, 5, 6])

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_newton_iter = 20

    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    # ocp.solver_options.nlp_solver_type = 'SQP'
    # ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
    # ocp.solver_options.nlp_solver_max_iter = 150

    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf

    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return acados_ocp_solver, acados_integrator
