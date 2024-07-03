from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from simple_model import simple_model
import scipy.linalg
import numpy as np
import math


def acados_settings(Tf, N):
    # create render arguments
    ocp = AcadosOcp()

    # export model
    model, constraint = simple_model()

    # define acados ODE
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
    model_ac.name = model.name
    ocp.model = model_ac

    # define constraint
    model_ac.con_h_expr = constraint.expr


    # dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx


    nsbx = 8
    nh = constraint.expr.shape[0]
    nsh = nh
    ns = nsh + nsbx
    

    ocp.cost.zl = 100 * np.ones((ns,))
    ocp.cost.zu = 100 * np.ones((ns,))
    ocp.cost.Zl = 1 * np.ones((ns,))
    ocp.cost.Zu = 1 * np.ones((ns,))


    # discretization
    ocp.dims.N = N

    
    # set cost
    Q = np.diag([ 1e-3, 1e-3, 1e-1, 1e-4, 1e-3, 0, 1e-20, 1e-20 ])

    R = np.eye(nu)
    R[0, 0] = 0
    R[1, 1] = 0


    Qe = Q#np.diag([ 0, 0, 1e-3, 1e-2, 1e-2, 1e-1, 1e-20, 1e-20])

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    unscale = N / Tf

    ocp.cost.W = unscale * scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Qe / unscale

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e


    # set intial references
    ocp.cost.yref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ocp.cost.yref_e = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    angle = 70*3.14156/180
    # setting constraints
    ocp.constraints.lbx = np.array([-1000, -1000,-1000, -10000000, -10000000, -angle, -800, -200])
    ocp.constraints.ubx = np.array([ 1000, 1000, 1000, 20 ,20, angle, 800, 200])
    ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5, 6, 7])   
    ocp.constraints.lbu = np.array([-50, -50 ])
    ocp.constraints.ubu = np.array([50, 50])
    ocp.constraints.idxbu = np.array([0, 1])   

    ocp.constraints.lh = np.array(
        [
            -1000,
        ]
    )
    ocp.constraints.uh = np.array(
        [
            2,
        ]
    )

    ocp.constraints.lsbx = np.zeros([nsbx])
    ocp.constraints.usbx = np.zeros([nsbx])
    ocp.constraints.idxsbx = np.array(range(nsbx))

    ocp.constraints.lsh = np.array([-50 ])
    ocp.constraints.ush = np.array([50 ])
    ocp.constraints.idxsh = np.array([0])


    # set intial condition
    ocp.constraints.x0 = model.x0

    #parameter initialization
    # ocp.parameter_values = np.array([0,0]) 
    # ocp.parameter_values = 

    # set QP solver and integration
    ocp.solver_options.tf = Tf
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    # ocp.solver_options.nlp_solver_step_length = 0.05
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.tol = 1e-4
    # ocp.solver_options.nlp_solver_tol_comp = 1e-1

    # create solver
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    return constraint, model, acados_solver
