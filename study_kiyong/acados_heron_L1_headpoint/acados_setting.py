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

    # Q_mat = 2*np.diag([1, 1, 20, 30, 30, 500, 1e-4, 1e-4])
    Q_mat = 2*np.diag([2, 2, 0, 3, 3, 1, 1e-4, 1e-4])
    R_mat = 2*np.diag([1e-3, 1e-3])

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat

    ocp.model.cost_y_expr = vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref  = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e, ))

    ocp.constraints.x0 = x0

    # set up parameters
    # oa = SX.sym('oa') 
    # ob = SX.sym('ob') 
    # oc = SX.sym('oc')
    # disturbance_u = SX.sym('disturbance_u')
    # disturbance_r = SX.sym('disturbance_r') 
    # p = vertcat(oa, ob, oc, disturbance_u, disturbance_r)
    # ocp.model.p = p 
    ocp.parameter_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 
    param = model.p

    #Normal boundary constraints
    h_expr = (model.x[0]*param[0]) + (model.x[1]*param[1]) + param[2]


    #Control Barrier Function
    M = 37.758 # Mass [kg]
    I = 18.35 # Inertial tensor [kg m^2]    
    Xu = 8.9149
    Xuu = 11.2101
    Nr = 16.9542
    Nrrr = 12.8966
    Yv = 15
    Yvv = 3
    Yr = 6
    Nv = 6
    dist = 0.3 # 30cm
    # Nominal CBF
    # h_expr = alpha*(param[0]*model.x[3]*cos(model.x[2]) + param[1]*model.x[3]*sin(model.x[2])) + (param[0]*model.x[0] + param[1]*model.x[1] + param[2])

    # Robust CLF
    # alpha1 = 0.5
    # alpha2 = 0.1
    # u_dot = ((model.x[5] + model.x[6]) - (Xu*model.x[3] + Xuu*model.x[3]*model.x[3]))/M  
    # x_dotdot = u_dot*cos(model.x[2]) - model.x[3]*model.x[4]*sin(model.x[2])
    # y_dotdot = u_dot*sin(model.x[2]) + model.x[3]*model.x[4]*cos(model.x[2])  
    # V = (param[0]*model.x[0] + param[1]*model.x[1] + param[2])**2
    # V_dot = 2*(param[0]*model.x[0] + param[1]*model.x[1] + param[2])*(param[0]*model.x[3]*cos(model.x[2]) + param[1]*model.x[3]*sin(model.x[2]))
    # V_dotdot = 2*(param[0]*model.x[3]*cos(model.x[2]) + param[1]*model.x[3]*sin(model.x[2]))**2 + 2*(param[0]*model.x[0] + param[1]*model.x[1] + param[2])*(param[0]*x_dotdot + param[1]*y_dotdot)
    # disturbance = 0.0#2*(param[0]*model.x[0] + param[1]*model.x[1] + param[2])*(param[0]*cos(model.x[2]) + param[1]*sin(model.x[2]))
    # h_expr = alpha2*alpha1*V_dotdot + (alpha2 + alpha1)*V_dot + V + alpha2*alpha1*((disturbance**2 + delta**2)**0.5 - delta)*d_u

    # Robust CBF
    # alpha1 = 5
    # alpha2 = 1
    # u_dot = ((model.x[5] + model.x[6]) - (Xu*model.x[3] + Xuu*model.x[3]*model.x[3]))/M  
    # x_dotdot = u_dot*cos(model.x[2]) - model.x[3]*model.x[4]*sin(model.x[2])
    # y_dotdot = u_dot*sin(model.x[2]) + model.x[3]*model.x[4]*cos(model.x[2])  
    # V = param[0]*model.x[0] + param[1]*model.x[1] + param[2]
    # V_dot = param[0]*model.x[3]*cos(model.x[2]) + param[1]*model.x[3]*sin(model.x[2])
    # V_dotdot = param[0]*x_dotdot + param[1]*y_dotdot
    # disturbance = (param[0]*cos(model.x[2]) + param[1]*sin(model.x[2]))
    # h_expr = alpha2*alpha1*V_dotdot + (alpha2 + alpha1)*V_dot + V - alpha2*alpha1*((disturbance**2 + delta**2)**0.5 - delta)*d_u



    ocp.model.con_h_expr = h_expr
 
    # For CBF
    ocp.constraints.uh = 1e10*np.ones(1)
    ocp.constraints.lh = -1e10*np.ones(1)

    # For CLF
    # ocp.constraints.uh = 0*np.ones(1)
    # ocp.constraints.lh = -1e10*np.ones(1)


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
    ocp.constraints.lbu = np.array([-Fmax/5,-Fmax/5])
    ocp.constraints.ubu = np.array([+Fmax/5,+Fmax/5])
    ocp.constraints.idxbu = np.array([0, 1])

    # ocp.constraints.lbx = np.array([-1, -1, -1, -Fmax, -Fmax])
    # ocp.constraints.ubx = np.array([2, 2, 1, Fmax, Fmax])
    ocp.constraints.lbx = np.array([-5, -5, -5, -Fmax, -Fmax])
    ocp.constraints.ubx = np.array([5, 5, 5, Fmax, Fmax])
    ocp.constraints.idxbx = np.array([3, 4, 5, 6, 7])

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