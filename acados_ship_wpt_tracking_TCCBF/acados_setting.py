from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from ship_model import export_ship_model
import scipy.linalg
import numpy as np
from casadi import SX, vertcat, cos, sin, sqrt
from load_param import load_ship_param

def setup_wpt_tracking(x0,mode):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()
    ship_p = load_ship_param

    # set model
    model = export_ship_model()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = ship_p.N
    # set cost module
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    Q_mat = ship_p.Q 
    R_mat = ship_p.R

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
    odx1 = SX.sym('odx1') 
    ody1 = SX.sym('ody1') 
    ox2 = SX.sym('ox2') 
    oy2 = SX.sym('oy2') 
    or2 = SX.sym('or2') 
    odx2 = SX.sym('odx2') 
    ody2 = SX.sym('ody2') 
    ox3 = SX.sym('ox3') 
    oy3 = SX.sym('oy3') 
    or3 = SX.sym('or3') 
    odx3 = SX.sym('odx3') 
    ody3 = SX.sym('ody3') 
    ox4 = SX.sym('ox4') 
    oy4 = SX.sym('oy4') 
    or4 = SX.sym('or4') 
    odx4 = SX.sym('odx4') 
    ody4 = SX.sym('ody4') 
    ox5 = SX.sym('ox5') 
    oy5 = SX.sym('oy5') 
    or5 = SX.sym('or5') 
    odx5 = SX.sym('odx5') 
    ody5 = SX.sym('ody5') 
    p = vertcat(ox1, oy1, or1, odx1, ody1, 
                ox2, oy2, or2, odx2, ody2, 
                ox3, oy3, or3, odx3, ody3, 
                ox4, oy4, or4, odx4, ody4, 
                ox5, oy5, or5, odx5, ody5)
    
    ocp.model.p = p 
    ocp.parameter_values = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 
                                     0.0, 0.0, 0.01, 0.0, 0.0, 
                                     0.0, 0.0, 0.01, 0.0, 0.0, 
                                     0.0, 0.0, 0.01, 0.0, 0.0, 
                                     0.0, 0.0, 0.01, 0.0, 0.0])

    num_obs = 5
    ocp.constraints.uh = 1e10 * np.ones(num_obs)
    ocp.constraints.lh = -1e-10 * np.ones(num_obs)
    h_expr = SX.zeros(num_obs,1)

    x0 = model.x[0]
    x1 = model.x[1]
    x2 = model.x[2]
    x3 = model.x[3]
    x0d = model.f_expl_expr[0]*ship_p.dt
    x1d = model.f_expl_expr[1]*ship_p.dt
    x2d = model.f_expl_expr[2]*ship_p.dt
    x3d = model.f_expl_expr[3]*ship_p.dt
    x0_n = x0+x0d
    x1_n = x1+x1d
    x2_n = x2+x2d
    x3_n = x3+x3d
    
    if ship_p.CBF == 0:

        ##################### MPC - Distance Constraints #############################
        h_expr[0] = (model.x[0]-ox1) ** 2 + (model.x[1] - oy1) ** 2 - or1**2
        h_expr[1] = (model.x[0]-ox2) ** 2 + (model.x[1] - oy2) ** 2 - or2**2
        h_expr[2] = (model.x[0]-ox3) ** 2 + (model.x[1] - oy3) ** 2 - or3**2
        h_expr[3] = (model.x[0]-ox4) ** 2 + (model.x[1] - oy4) ** 2 - or4**2
        h_expr[4] = (model.x[0]-ox5) ** 2 + (model.x[1] - oy5) ** 2 - or5**2

    if ship_p.CBF == 1:
    
        ##################### MPC - Euclidean Distance-based CBF #####################
        for i in range(num_obs):
            ox = p[5*i+0]
            oy = p[5*i+1]
            obr = p[5*i+2]
            odx = p[5*i+3]*ship_p.dt
            ody = p[5*i+4]*ship_p.dt
            ox_n = ox + odx
            oy_n = oy + ody

            B = np.sqrt( (x0-ox)**2 + (x1 - oy)**2) - obr
            Bdot = ((x0-ox)*x3*cos(x2) + (x1-oy)*x3*sin(x2))/np.sqrt((x0-ox)**2 + (x1 - oy)**2)
            hk = Bdot + ship_p.gamma1/(obr+0.0001)*B
            B = np.sqrt( (x0_n-ox_n)**2 + (x1_n - oy_n)**2) - obr
            Bdot = ((x0_n-ox_n)*(x3_n)*cos(x2_n) + (x1_n-oy_n)*(x3_n)*sin(x2_n))/np.sqrt((x0_n-ox_n)**2 + (x1_n - oy_n)**2)
            hkn = Bdot + ship_p.gamma1/(obr+0.0001)*B
            h_expr[i] = hkn - (1-ship_p.gamma2)*hk
            # h_expr[i] = hk

    if ship_p.CBF == 2:
        
        ##################### MPC - TC Distance-based CBF #####################
        for i in range(num_obs):
            ox = p[5*i+0]
            oy = p[5*i+1]
            obr = p[5*i+2]
            odx = p[5*i+3]*ship_p.dt
            ody = p[5*i+4]*ship_p.dt
            ox_n = ox + odx
            oy_n = oy + ody

            R = x3/ship_p.rmax*ship_p.gamma_TC1
            B1 = np.sqrt( (ox-x0-R*cos(x2-np.pi/2))**2 + (oy-x1-R*sin(x2-np.pi/2))**2) - (obr+R)
            B2 = np.sqrt( (ox-x0-R*cos(x2+np.pi/2))**2 + (oy-x1-R*sin(x2+np.pi/2))**2) - (obr+R)
            if ship_p.TCCBF == 1:
                hk = B1
            if ship_p.TCCBF == 2:
                hk = B2
            if ship_p.TCCBF == 3:
                hk = np.log((np.exp(B1)+np.exp(B2)-1))
            
            ## Crossing
            if mode == 'crossing':
                if i == 2 or i ==4:
                    hk = B2

            R = x3_n/ship_p.rmax*ship_p.gamma_TC1
            B1 = np.sqrt( (ox_n-x0_n-R*cos(x2_n-np.pi/2))**2 + (oy_n-x1_n-R*sin(x2_n-np.pi/2))**2) - (obr+R)
            B2 = np.sqrt( (ox_n-x0_n-R*cos(x2_n+np.pi/2))**2 + (oy_n-x1_n-R*sin(x2_n+np.pi/2))**2) - (obr+R)
            if ship_p.TCCBF == 1:
                hkn = B1
            if ship_p.TCCBF == 2:
                hkn = B2
            if ship_p.TCCBF == 3:
                hkn = np.log((np.exp(B1)+np.exp(B2)-1))
            
            ## Crossing
            if mode == 'crossing':
                if i == 2 or i ==4:
                    hkn = B2
            
            h_expr[i] = hkn - (1-ship_p.gamma_TC2)*hk

    

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
    ocp.constraints.lbu = np.array([-ship_p.dFxmax,-ship_p.dFnmax])
    ocp.constraints.ubu = np.array([+ship_p.dFxmax,+ship_p.dFnmax])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([0, -1, ship_p.Fxmin, ship_p.Fnmin])
    ocp.constraints.ubx = np.array([3, 1, ship_p.Fxmax, ship_p.Fnmax])
    ocp.constraints.idxbx = np.array([3, 4, 5, 6])

    ocp.solver_options.print_level = 0
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    # ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    # ocp.solver_options.integrator_type = 'DISCRETE'
    # ocp.solver_options.tol = 1e-10
    # ocp.solver_options.sim_method_newton_iter = 250
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    # ocp.solver_options.qp_solver_cond_N = int(ship_p.N)

    # set prediction horizon
    ocp.solver_options.tf = int(ship_p.dt*ship_p.N)

    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return acados_ocp_solver, acados_integrator


