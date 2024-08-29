import numpy as np
from scipy.optimize import minimize

def CLF_QP(state,param, weight, lamda):

    
    # Define the objective function with parameters
    def objective(x):
        return weight[0] * (x[0])**2 + weight[1] * (x[1])**2  + weight[2] * (x[2])**2 
    

    # Define the inequality constraint with parameters
    def head_ineq_constraint(x, state, param):

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

        head_xn = xn + head_dist*np.cos(psi)
        head_yn = yn + head_dist*np.sin(psi)

        u_dot = ((x[0] + x[1]) - Xu*u - Xuu*np.sqrt(u*u)*u)/M
        v_dot = (-Yv*v - Yvv*np.sqrt(v*v)*v - Yr*r)/M
        r_dot = ((-x[0]+x[1])*dist - Nr*r - Nrrr*r*r*r - Nv*v)/I

        head_xn_dot = u*np.cos(psi) - v*np.sin(psi) - head_dist*r*np.sin(psi)
        head_yn_dot = u*np.sin(psi) + v*np.cos(psi) + head_dist*r*np.cos(psi)

        head_xn_dotdot = u_dot*np.cos(psi) - u*r*np.sin(psi) - v_dot*np.sin(psi) - v*r*np.cos(psi) - head_dist*r_dot*np.sin(psi) - head_dist*r*r*np.cos(psi)
        head_yn_dotdot = u_dot*np.sin(psi) + u*r*np.cos(psi) + v_dot*np.cos(psi) - v*r*np.sin(psi) + head_dist*r_dot*np.cos(psi) - head_dist*r*r*np.sin(psi)

        V = (param[0] - head_xn)**2 + (param[1] - head_yn)**2 
        V_dot = 2*(head_xn - param[0])*head_xn_dot + 2*(head_yn - param[1])*head_yn_dot
        V_dotdot =  2*(head_xn - param[0])*head_xn_dotdot + 2*(head_yn - param[1])*head_yn_dotdot + 2*head_xn_dot**2 + 2*head_yn_dot**2

        print("Head : ",V)
        return -(V + (lamda[0] + lamda[1])*V_dot + lamda[0]*lamda[1]*V_dotdot + x[2]) 



    x0 = [0, 0, 0]

    # Define the bounds
    bounds = [(-25, 25), (-25, 25), (None, None)]  # Bounds for x0, x1, and x2

    # Define the constraints
    # eq_constraints = [example_eq_constraint]
    ineq_constraints = [head_ineq_constraint]

    # Combine equality and inequality constraints if any are provided
    constraints = []
    
    # if eq_constraints is not None:
    #     for constraint in eq_constraints:
    #         constraints.append({
    #             'type': 'eq',
    #             'fun': lambda x, constraint=constraint: constraint(x, params)
    #         })
    
    if ineq_constraints is not None:
        for constraint in ineq_constraints:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, constraint=constraint: constraint(x, state, param)
            })
    
    # Perform the optimization using Sequential Least Squares Programming (SLSQP)
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    


    return result.x

