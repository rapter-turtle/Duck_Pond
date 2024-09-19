from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from heron_model import export_heron_model
import scipy.linalg
import numpy as np
from casadi import SX, vertcat, cos, sin

def setup_trajectory_tracking(x0, Fmax, N_horizon, Tf):
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

    Q_mat = 2*np.diag([5, 5, 20, 30, 50, 1e-4, 1e-4])
    R_mat = 2*np.diag([1e-3, 1e-3])

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
    h_expr[0] = (model.x[0]-ox1) ** 2 + (model.x[1] - oy1) ** 2 - or1**2
    h_expr[1] = (model.x[0]-ox2) ** 2 + (model.x[1] - oy2) ** 2 - or2**2
    h_expr[2] = (model.x[0]-ox3) ** 2 + (model.x[1] - oy3) ** 2 - or3**2
    h_expr[3] = (model.x[0]-ox4) ** 2 + (model.x[1] - oy4) ** 2 - or4**2
    h_expr[4] = (model.x[0]-ox5) ** 2 + (model.x[1] - oy5) ** 2 - or5**2
    ocp.model.con_h_expr = h_expr

    ocp.constraints.idxsh = np.array([0,1,2,3,4])
    ocp.constraints.idxsh_e = np.array([0,1,2,3,4])
    Zh = 1e5 * np.ones(num_obs)
    zh = 1e5 * np.ones(num_obs)
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

    # Q_mat = 2*np.diag([5, 5, 10, 15, 15, 50, 1e-4, 1e-4])
    Q_mat = 2*np.diag([4, 4, 40, 20, 20, 50, 1e-4, 1e-4])
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
    # h_expr = (model.x[0]*param[0]) + (model.x[1]*param[1]) + param[2]


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

        # xr = (x_head - xt)*np.cos(tpsi) - (y_head - yt)*np.sin(tpsi)
        # yr = (x_head - xt)*np.sin(tpsi) + (y_head - yt)*np.cos(tpsi)

        # xr_dot = x_head_dot*np.cos(tpsi) - y_head_dot*np.sin(tpsi)
        # yr_dot = x_head_dot*np.sin(tpsi) + y_head_dot*np.cos(tpsi)
        # xr_dotdot = x_head_dotdot*np.cos(tpsi) - y_head_dotdot*np.sin(tpsi)
        # yr_dotdot = x_head_dotdot*np.sin(tpsi) + y_head_dotdot*np.cos(tpsi)

    # u_dot = ((model.x[6] + model.x[7]) - Xu*model.x[3] - Xuu*np.sqrt(model.x[3]*model.x[3])*model.x[3])/M 
    # v_dot = (-Yv*model.x[4] - Yvv*np.sqrt(model.x[4]*model.x[4])*model.x[4] - Yr*model.x[5])/M 
    # r_dot = ((-model.x[6]+model.x[7])*dist - Nr*model.x[5] - Nrrr*model.x[5]*model.x[5]*model.x[5] - Nv*model.x[4])/I 

    u_dot = ((model.x[6] + model.x[7]) - Xu*model.x[3] - Xuu*np.sqrt(model.x[3]*model.x[3])*model.x[3])/M + param[0]
    v_dot = (-Yv*model.x[4] - Yvv*np.sqrt(model.x[4]*model.x[4])*model.x[4] - Yr*model.x[5])/M + param[1]
    r_dot = ((-model.x[6]+model.x[7])*dist - Nr*model.x[5] - Nrrr*model.x[5]*model.x[5]*model.x[5] - Nv*model.x[4])/I + param[2] 

    xt = 10.0
    yt = 10.0
    tpsi = -0.1
    beta = 7
    x_head = model.x[0] + cos(model.x[2])
    y_head = model.x[1] + sin(model.x[2])
    x_head_dot = model.x[3]*cos(model.x[2]) - model.x[4]*sin(model.x[2]) - model.x[5]*sin(model.x[2]) 
    y_head_dot = model.x[3]*sin(model.x[2]) + model.x[4]*cos(model.x[2]) + model.x[5]*cos(model.x[2])    
    x_head_dotdot = u_dot*np.cos(model.x[2]) - v_dot*np.sin(model.x[2]) - r_dot*np.sin(model.x[2]) - model.x[3]*model.x[5]*np.sin(model.x[2]) - model.x[4]*model.x[5]*np.cos(model.x[2]) - model.x[5]*model.x[5]*np.cos(model.x[2])
    y_head_dotdot = u_dot*np.sin(model.x[2]) + v_dot*np.cos(model.x[2]) + r_dot*np.cos(model.x[2]) + model.x[3]*model.x[5]*np.cos(model.x[2]) - model.x[4]*model.x[5]*np.sin(model.x[2]) - model.x[5]*model.x[5]*np.sin(model.x[2])

    xr = (x_head - xt)*np.cos(tpsi) - (y_head - yt)*np.sin(tpsi)
    yr = (x_head - xt)*np.sin(tpsi) + (y_head - yt)*np.cos(tpsi)
    xr_dot = x_head_dot*np.cos(tpsi) - y_head_dot*np.sin(tpsi)
    yr_dot = x_head_dot*np.sin(tpsi) + y_head_dot*np.cos(tpsi)
    xr_dotdot = x_head_dotdot*np.cos(tpsi) - y_head_dotdot*np.sin(tpsi)
    yr_dotdot = x_head_dotdot*np.sin(tpsi) + y_head_dotdot*np.cos(tpsi)

    a = 0.5
    V = -(beta*yr*yr)/(1+yr*yr) - xr
    V_dot = -(2*beta*yr*yr_dot)/(1+yr*yr)**2 - xr_dot
    V_dotdot = ((-2 * beta * (1 + yr*yr) + 4*beta*yr*yr)*yr_dot*yr_dot - 2 * beta * yr*yr_dotdot*(1 +yr**2)) / (1 + yr**2)**3 - xr_dotdot
    h_expr = V_dotdot + (0.1 + a)*V_dot + 0.1*a*V
    # h_expr = 0.1*V + V_dot


    ocp.model.con_h_expr = h_expr
 
    # For CBF
    ocp.constraints.uh = 1e10*np.ones(1)
    # ocp.constraints.lh = -1e10*np.ones(1)
    ocp.constraints.lh = 0*np.ones(1)

    # For CLF
    # ocp.constraints.uh = 0*np.ones(1)
    # ocp.constraints.lh = -1e10*np.ones(1)


    ocp.constraints.idxsh = np.array([0])
    ocp.constraints.idxsh_e = np.array([0])
    Zh = 1e7*np.ones(1)#1e1*np.ones(1)
    zh = 1e7*np.ones(1)#1e1*np.ones(1)
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
    ocp.constraints.lbx = np.array([-5, -5, -5, -10*Fmax, -10*Fmax])
    ocp.constraints.ubx = np.array([5, 5, 5, 10*Fmax, 10*Fmax])
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
