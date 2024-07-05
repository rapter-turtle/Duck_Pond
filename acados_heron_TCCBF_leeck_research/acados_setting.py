from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from heron_model import export_heron_model
import scipy.linalg
import numpy as np
from casadi import SX, vertcat, cos, sin, sqrt

def setup_trajectory_tracking(x0, Fmax, N_horizon, Tf, dt):
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

    Q_mat = 2*np.diag([2, 2, 20, 100, 5, 1e-4, 1e-4])
    R_mat = 2*np.diag([1e-4, 1e-4])

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat

    ocp.model.cost_y_expr = vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref  = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e, ))

    ocp.constraints.x0 = x0

    # set up parameters
    ox1 = SX.sym('ox1') 
    oy1 = SX.sym('oy1') 
    or1 = SX.sym('or1') 
    ox2 = SX.sym('ox2') 
    oy2 = SX.sym('oy2') 
    or2 = SX.sym('or2') 
    ox3 = SX.sym('ox3') 
    oy3 = SX.sym('oy3') 
    or3 = SX.sym('or3') 
    ox4 = SX.sym('ox4') 
    oy4 = SX.sym('oy4') 
    or4 = SX.sym('or4') 
    ox5 = SX.sym('ox5') 
    oy5 = SX.sym('oy5') 
    or5 = SX.sym('or5') 
    p = vertcat(ox1, oy1, or1, 
                ox2, oy2, or2, 
                ox3, oy3, or3, 
                ox4, oy4, or4, 
                ox5, oy5, or5)
    
    ocp.model.p = p 
    ocp.parameter_values = np.array([0.0, 0.0, 0.0, 
                                     0.0, 0.0, 0.0, 
                                     0.0, 0.0, 0.0, 
                                     0.0, 0.0, 0.0, 
                                     0.0, 0.0, 0.0])

    num_obs = 5
    ocp.constraints.uh = 1e10 * np.ones(num_obs)
    ocp.constraints.lh = np.zeros(num_obs)
    h_expr = SX.zeros(num_obs,1)
    
    ## MPC - Distance Constraints
    # h_expr[0] = (model.x[0]-ox1) ** 2 + (model.x[1] - oy1) ** 2 - or1**2
    # h_expr[1] = (model.x[0]-ox2) ** 2 + (model.x[1] - oy2) ** 2 - or2**2
    # h_expr[2] = (model.x[0]-ox3) ** 2 + (model.x[1] - oy3) ** 2 - or3**2
    # h_expr[3] = (model.x[0]-ox4) ** 2 + (model.x[1] - oy4) ** 2 - or4**2
    # h_expr[4] = (model.x[0]-ox5) ** 2 + (model.x[1] - oy5) ** 2 - or5**2

    ## MPC - Euclidean Distance-based CBF
    gamma1 = 0.4
    gamma2 = 0.2

    x0d = model.f_expl_expr[0]*dt
    x1d = model.f_expl_expr[1]*dt
    x2d = model.f_expl_expr[2]*dt
    x3d = model.f_expl_expr[3]*dt
    
    B = np.sqrt( (model.x[0]-ox1) ** 2 + (model.x[1] - oy1) ** 2) - or1
    Bdot = ((model.x[0]-ox1)*model.x[3]*cos(model.x[2]) + (model.x[1]-oy1)*model.x[3]*sin(model.x[2]))/np.sqrt((model.x[0]-ox1)**2 + (model.x[1] - oy1) ** 2)
    hk = Bdot + gamma1*B
    B = np.sqrt( (model.x[0]+x0d-ox1) ** 2 + (model.x[1]+x1d - oy1) ** 2) - or1
    Bdot = ((model.x[0]+x0d-ox1)*(model.x[3]+x3d)*cos(model.x[2]+x2d) + (model.x[1]+x1d-oy1)*(model.x[3]+x3d)*sin(model.x[2]+x2d))/np.sqrt((model.x[0]+x0d-ox1)**2 + (model.x[1]+x1d - oy1) ** 2)
    hkn = Bdot + gamma1*B
    h_expr[0] = hkn - (1-gamma2)*hk


    B = np.sqrt( (model.x[0]-ox2) ** 2 + (model.x[1] - oy2) ** 2) - or2
    Bdot = ((model.x[0]-ox2)*model.x[3]*cos(model.x[2]) + (model.x[1]-oy2)*model.x[3]*sin(model.x[2]))/np.sqrt((model.x[0]-ox2)**2 + (model.x[1] - oy2) ** 2)
    hk = Bdot + gamma1*B    
    B = np.sqrt( (model.x[0]+x0d-ox2) ** 2 + (model.x[1]+x1d - oy2) ** 2) - or2
    Bdot = ((model.x[0]+x0d-ox2)*(model.x[3]+x3d)*cos(model.x[2]+x2d) + (model.x[1]+x1d-oy2)*(model.x[3]+x3d)*sin(model.x[2]+x2d))/np.sqrt((model.x[0]+x0d-ox2)**2 + (model.x[1]+x1d - oy2) ** 2)
    hkn = Bdot + gamma1*B
    h_expr[1] = hkn - (1-gamma2)*hk

    B = np.sqrt( (model.x[0]-ox3) ** 2 + (model.x[1] - oy3) ** 2) - or3
    Bdot = ((model.x[0]-ox3)*model.x[3]*cos(model.x[2]) + (model.x[1]-oy3)*model.x[3]*sin(model.x[2]))/np.sqrt((model.x[0]-ox3)**2 + (model.x[1] - oy3) ** 2)
    hk = Bdot + gamma1*B
    B = np.sqrt( (model.x[0]+x0d-ox3) ** 2 + (model.x[1]+x1d - oy3) ** 2) - or3
    Bdot = ((model.x[0]+x0d-ox3)*(model.x[3]+x3d)*cos(model.x[2]+x2d) + (model.x[1]+x1d-oy3)*(model.x[3]+x3d)*sin(model.x[2]+x2d))/np.sqrt((model.x[0]+x0d-ox3)**2 + (model.x[1]+x1d - oy3) ** 2)
    hkn = Bdot + gamma1*B
    h_expr[2] = hkn - (1-gamma2)*hk

    B = np.sqrt( (model.x[0]-ox4) ** 2 + (model.x[1] - oy4) ** 2) - or4
    Bdot = ((model.x[0]-ox4)*model.x[3]*cos(model.x[2]) + (model.x[1]-oy4)*model.x[3]*sin(model.x[2]))/np.sqrt((model.x[0]-ox4)**2 + (model.x[1] - oy4) ** 2)
    hk = Bdot + gamma1*B
    B = np.sqrt( (model.x[0]+x0d-ox4) ** 2 + (model.x[1]+x1d - oy4) ** 2) - or4
    Bdot = ((model.x[0]+x0d-ox4)*(model.x[3]+x3d)*cos(model.x[2]+x2d) + (model.x[1]+x1d-oy4)*(model.x[3]+x3d)*sin(model.x[2]+x2d))/np.sqrt((model.x[0]+x0d-ox4)**2 + (model.x[1]+x1d - oy4) ** 2)
    hkn = Bdot + gamma1*B
    h_expr[3] = hkn - (1-gamma2)*hk

    B = np.sqrt( (model.x[0]-ox5) ** 2 + (model.x[1] - oy5) ** 2) - or5
    Bdot = ((model.x[0]-ox5)*model.x[3]*cos(model.x[2]) + (model.x[1]-oy5)*model.x[3]*sin(model.x[2]))/np.sqrt((model.x[0]-ox5)**2 + (model.x[1] - oy5) ** 2)
    hk = Bdot + gamma1*B
    B = np.sqrt( (model.x[0]+x0d-ox5) ** 2 + (model.x[1]+x1d - oy5) ** 2) - or5
    Bdot = ((model.x[0]+x0d-ox5)*(model.x[3]+x3d)*cos(model.x[2]+x2d) + (model.x[1]+x1d-oy5)*(model.x[3]+x3d)*sin(model.x[2]+x2d))/np.sqrt((model.x[0]+x0d-ox5)**2 + (model.x[1]+x1d - oy5) ** 2)
    hkn = Bdot + gamma1*B
    h_expr[4] = hkn - (1-gamma2)*hk


    # h_expr[0] = (model.x[0]-ox1) ** 2 + (model.x[1] - oy1) ** 2 - or1**2
    # h_expr[1] = (model.x[0]-ox2) ** 2 + (model.x[1] - oy2) ** 2 - or2**2
    # h_expr[2] = (model.x[0]-ox3) ** 2 + (model.x[1] - oy3) ** 2 - or3**2
    # h_expr[3] = (model.x[0]-ox4) ** 2 + (model.x[1] - oy4) ** 2 - or4**2
    # h_expr[4] = (model.x[0]-ox5) ** 2 + (model.x[1] - oy5) ** 2 - or5**2



    ocp.model.con_h_expr = h_expr

    ocp.constraints.idxsh = np.array([0,1,2,3,4])
    ocp.constraints.idxsh_e = np.array([0,1,2,3,4])
    Zh = 1e6 * np.ones(num_obs)
    zh = 1e6 * np.ones(num_obs)
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
    ocp.constraints.lbu = np.array([-Fmax/5,-Fmax/5])
    ocp.constraints.ubu = np.array([+Fmax/5,+Fmax/5])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([-1, -1, -Fmax, -Fmax])
    ocp.constraints.ubx = np.array([2, 1, Fmax, Fmax])
    ocp.constraints.idxbx = np.array([3, 4, 5, 6])

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
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


