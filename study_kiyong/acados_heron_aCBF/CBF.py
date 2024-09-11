import numpy as np
from scipy.optimize import minimize

def CBF_QP(state, control_input, param, weight, ll, param_estim):

    # Define the objective function with parameters
    def objective(x):
        # return weight[0] * (x[0] - control_input[0])**2 + weight[1] * (x[1] - control_input[1])**2  + weight[2] * (x[2])**2   
        return weight[2] * (x[2])**2   

    # Define the inequality constraint with parameters
    def head_ineq_constraint(x, state, param, param_estim):

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
        
        ll = np.array([0.5,0.5])
        u_dot = ((x[0] + x[1]) - Xu*u - Xuu*np.sqrt(u*u)*u)/M
        v_dot = (-Yv*v - Yvv*np.sqrt(v*v)*v - Yr*r)/M
        r_dot = ((-x[0]+x[1])*dist - Nr*r - Nrrr*r*r*r - Nv*v)/I


        V = param[0]*(xn + head_dist*np.cos(psi)) + param[1]*(yn + head_dist*np.sin(psi)) + param[2]
        V_dot = param[0]*(u*np.cos(psi) - v*np.sin(psi) - r*head_dist*np.sin(psi)) + param[1]*(u*np.sin(psi) + v*np.cos(psi) + r*head_dist*np.cos(psi))
        V_dotdot = param[0]*(u_dot*np.cos(psi) - u*r*np.sin(psi) - v_dot*np.sin(psi) - v*r*np.cos(psi) - r_dot*head_dist*np.sin(psi) - r*r*head_dist*np.cos(psi)) + param[1]*(u_dot*np.sin(psi) + u*r*np.cos(psi) + v_dot*np.cos(psi) - v*r*np.sin(psi) + r_dot*head_dist*np.cos(psi) - r*r*head_dist*np.sin(psi))
  
        print(V_dotdot + (ll[0] + ll[1])*V_dot + ll[0]*ll[1]*V)
        print(V_dot + ll[0]*V)
        print(V)
        return (V_dotdot + (ll[0] + ll[1])*V_dot + ll[0]*ll[1]*V + x[2])# Example inequality constraint



    x0 = [control_input[0], control_input[1], 0]

    # Define the bounds
    bounds = [(-30, 30), (-30, 30), (-0.1, 0.1)]  # Bounds for x0, x1, x2, and x3

    # Define the constraints
    # ineq_constraints = [head_ineq_constraint, tail_ineq_constraint]

    ineq_constraints = [head_ineq_constraint]
    # Combine equality and inequality constraints if any are provided
    constraints = []
    
    if ineq_constraints is not None:
        for constraint in ineq_constraints:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, constraint=constraint, state=state, param=param, param_estim=param_estim: constraint(x, state, param, param_estim)
            })
    
    # Perform the optimization using Sequential Least Squares Programming (SLSQP)
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    print(result.x[2])

    return result.x
