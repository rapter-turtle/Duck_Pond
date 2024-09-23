import numpy as np
from scipy.optimize import minimize

def CBF_QP(state, control_input, param, weight, ll, param_estim, x_tship):

    # Define the objective function with parameters
    def objective(x):
        return weight[0] * (x[0] - control_input[0])**2 + weight[1] * (x[1] - control_input[1])**2    
        # return weight[2] * (x[2])**2   
        # return weight[0] * (x[0] - control_input[0])**2 + weight[1] * (x[1] - control_input[1])**2  + weight[2] * (x[2])**2   
        
    # Define the inequality constraint with parameters
    def head_ineq_constraint(x, state, param, param_estim, x_tship):

        # constants
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
        head_dist = 1.0

        u = state[3]
        v = state[4]
        r = state[5]
        xn = state[0]
        yn = state[1]
        psi = state[2]

        xt = x_tship[0]- 2.5*np.cos(x_tship[2] + 3.141592*0.5)
        yt = x_tship[1] - 2.5*np.sin(x_tship[2] + 3.141592*0.5)
        tpsi = x_tship[2]

        beta = 1
        
        ll = np.array([0.1,0.1])

        u_dot = ((x[0] + x[1]) - Xu*u - Xuu*np.sqrt(u*u)*u)/M + param_estim[0]
        v_dot = (-Yv*v - Yvv*np.sqrt(v*v)*v - Yr*r)/M + param_estim[1]
        r_dot = ((-x[0]+x[1])*dist - Nr*r - Nrrr*r*r*r - Nv*v)/I + param_estim[2]

        x_head = xn + head_dist*np.cos(psi)
        y_head = yn + head_dist*np.sin(psi)
        x_head_dot = u*np.cos(psi) - v*np.sin(psi) - r*head_dist*np.sin(psi)
        y_head_dot = u*np.sin(psi) + v*np.cos(psi) + r*head_dist*np.cos(psi)
        x_head_dotdot = u_dot*np.cos(psi) - v_dot*np.sin(psi) - r_dot*head_dist*np.sin(psi) - u*r*np.sin(psi) - v*r*np.cos(psi) - head_dist*r*r*np.cos(psi)
        y_head_dotdot = u_dot*np.sin(psi) + v_dot*np.cos(psi) + r_dot*head_dist*np.cos(psi) + u*r*np.cos(psi) - v*r*np.sin(psi) - head_dist*r*r*np.sin(psi)

        xr = (x_head - xt)*np.cos(tpsi) - (y_head - yt)*np.sin(tpsi)
        yr = (x_head - xt)*np.sin(tpsi) + (y_head - yt)*np.cos(tpsi)

        xr_dot = x_head_dot*np.cos(tpsi) - y_head_dot*np.sin(tpsi)
        yr_dot = x_head_dot*np.sin(tpsi) + y_head_dot*np.cos(tpsi)
        xr_dotdot = x_head_dotdot*np.cos(tpsi) - y_head_dotdot*np.sin(tpsi)
        yr_dotdot = x_head_dotdot*np.sin(tpsi) + y_head_dotdot*np.cos(tpsi)



        V = -(beta*yr*yr)/(1+yr*yr) - xr + 1
        V_dot = -(2*beta*yr*yr_dot)/(1+yr*yr)**2 - xr_dot
        # V_dotdot = (-2 * beta * yr_dot**2 + 2 * beta * yr**2 * yr_dot**2 - 2 * beta * yr * yr_dotdot * (1 +yr**2)) / (1 + yr**2)**2 - xr_dotdot
        V_dotdot = ((-2 * beta * (1 + yr*yr) + 4*beta*yr*yr)*yr_dot*yr_dot - 2 * beta * yr*yr_dotdot*(1 +yr**2)) / (1 + yr**2)**3 - xr_dotdot
        # if (V_dotdot + (ll[0] + ll[1])*V_dot + ll[0]*ll[1]*V) < -0.5:
        #     print("real : ",(V_dotdot + (ll[0] + ll[1])*V_dot + ll[0]*ll[1]*V))
        # if (V_dotdot + (ll[0] + ll[1])*V_dot + ll[0]*ll[1]*V + x[2]) < -0.5:
        #     print((V_dotdot + (ll[0] + ll[1])*V_dot + ll[0]*ll[1]*V + x[2]))    
        return (V_dotdot + (ll[0] + ll[1])*V_dot + ll[0]*ll[1]*V)# Example inequality constraint
        # return (V_dotdot + (ll[0] + ll[1])*V_dot + ll[0]*ll[1]*V + x[2])# Example inequality constraint


    x0 = [control_input[0], control_input[1], 0]

    # Define the bounds
    bounds = [(-60, 60), (-60, 60), (-0.1, 0.1)]  # Bounds for x0, x1, x2, and x3

    # Define the constraints
    # ineq_constraints = [head_ineq_constraint, tail_ineq_constraint]

    ineq_constraints = [head_ineq_constraint]
    # Combine equality and inequality constraints if any are provided
    constraints = []
    
    if ineq_constraints is not None:
        for constraint in ineq_constraints:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, constraint=constraint, state=state, param=param, param_estim=param_estim: constraint(x, state, param, param_estim, x_tship)
            })
    
    # Perform the optimization using Sequential Least Squares Programming (SLSQP)
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    print(result.x[2])
    # if np.abs(result.x[2]) > 0.1:
    #     print("damn",result.x[2])
    return result.x
