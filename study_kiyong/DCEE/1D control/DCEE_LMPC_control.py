import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# True dynamics coefficients (unknown to the controller)
true_Cxx, true_Cxy, true_Cyy, true_Cyx = 3.0, 2.0, 5.0, 1.5

# Initial Bayesian priors for the parameters
Cxx_range = np.linspace(0.1, 10.0, 50)
Cxy_range = np.linspace(0.1, 10.0, 50)
Cyy_range = np.linspace(0.1, 10.0, 50)
Cyx_range = np.linspace(0.1, 10.0, 50)

# Gaussian prior distributions centered around initial estimates with wider variance
Cxx_prior = np.exp(-0.5 * ((Cxx_range - 1.0) / 1.0) ** 2)
Cxy_prior = np.exp(-0.5 * ((Cxy_range - 1.0) / 1.0) ** 2)
Cyy_prior = np.exp(-0.5 * ((Cyy_range - 1.0) / 1.0) ** 2)
Cyx_prior = np.exp(-0.5 * ((Cyx_range - 1.0) / 1.0) ** 2)

# Normalize priors
Cxx_prior /= Cxx_prior.sum()
Cxy_prior /= Cxy_prior.sum()
Cyy_prior /= Cyy_prior.sum()
Cyx_prior /= Cyx_prior.sum()

# Likelihood function for Bayesian inference
def likelihood(observed, predicted, sigma=0.5):
    return np.exp(-0.5 * ((observed - predicted) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

# Discretize system parameters
dt = 0.1
A = np.array([
    [0, 1, 0, 0],
    [0, -true_Cxx, 0, -true_Cxy],
    [0, 0, 0, 1],
    [0, -true_Cyx, 0, -true_Cyy]
])
B = np.array([[0], [1], [0], [0]])
A_d = np.eye(4) + A * dt
B_d = B * dt

# Cost parameters
alpha = 0.1  # Regularization parameter for control effort
beta = 0.0  # Weight for covariance trace in cost
gamma = 10000  # Weight for variance in state prediction cost

# Simulation parameters
t_final = 200
n_steps = int(t_final / dt)
N = 10  # MPC horizon

# Initial state
x = np.array([0, 0, 0, 0])
x_history = [x]
time = [0]
u_history = []
target_y_values = []

# Lists to store posterior means and trace of covariance for plotting
mean_Cxx_values, mean_Cxy_values, mean_Cyy_values, mean_Cyx_values = [], [], [], []
cov_trace_values = []

# Loop through each time step to compute control and update state with Bayesian inference
for i in range(n_steps):
    # Update the time-varying target
    current_time = i * dt
    target_y = np.sin(0.2 * np.sin(current_time * 0.1)) + 1
    target_y_values.append(target_y)

    # Calculate posterior mean of Cxx, Cxy, Cyy, and Cyx values
    mean_Cxx = (Cxx_range * Cxx_prior).sum()
    mean_Cxy = (Cxy_range * Cxy_prior).sum()
    mean_Cyy = (Cyy_range * Cyy_prior).sum()
    mean_Cyx = (Cyx_range * Cyx_prior).sum()
    
    # Store posterior means for plotting
    mean_Cxx_values.append(mean_Cxx)
    mean_Cxy_values.append(mean_Cxy)
    mean_Cyy_values.append(mean_Cyy)
    mean_Cyx_values.append(mean_Cyx)

    # Compute the variance for each parameter to calculate the trace of the covariance matrix
    var_Cxx = ((Cxx_range - mean_Cxx) ** 2 * Cxx_prior).sum()
    var_Cxy = ((Cxy_range - mean_Cxy) ** 2 * Cxy_prior).sum()
    var_Cyy = ((Cyy_range - mean_Cyy) ** 2 * Cyy_prior).sum()
    var_Cyx = ((Cyx_range - mean_Cyx) ** 2 * Cyx_prior).sum()
    cov_trace = var_Cxx + var_Cxy + var_Cyy + var_Cyx
    cov_trace_values.append(cov_trace)

    # Define MPC optimization variables
    u = cp.Variable(N)
    x_pred = x  # Initialize predicted state for the horizon
    cost = 0
    constraints = []

    # Build the MPC cost function over the prediction horizon
    for t in range(N):
        # Update the A_d matrix based on the estimated parameters
        A_d[1, 1] = -mean_Cxx
        A_d[1, 3] = -mean_Cxy
        A_d[3, 1] = -mean_Cyx
        A_d[3, 3] = -mean_Cyy

        # Predict the next state using the discretized dynamics
        x_next = A_d @ x_pred + B_d.flatten() * u[t]
        xdot_next = x_next[1]
        ydot_next = x_next[3]

        # Variance of xdot_next and ydot_next
        var_xdot_next = xdot_next**2 * var_Cxx + ydot_next**2 * var_Cxy
        var_ydot_next = ydot_next**2 * var_Cyy + xdot_next**2 * var_Cyx

        # Accumulate cost: target tracking, control effort, and variance penalty
        target = target_y  # Assume constant target within horizon for simplicity
        cost += 1 * cp.square(x_next[2] - target) + alpha * cp.square(u[t]) + gamma * (var_xdot_next + var_ydot_next)

        # Update the predicted state for the next step in the horizon
        x_pred = x_next

        # Add control input constraints
        constraints += [-50000 <= u[t], u[t] <= 50000]

    # Add the trace of covariance to the total cost
    cost += beta * cov_trace

    # Formulate the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.SCS)

    # Retrieve the optimal control input for the first step in the horizon
    u_opt = u.value[0] if u.value is not None else 0  # Set to 0 if no solution

    # Update state using the computed optimal control with the discretized dynamics
    x = A_d @ x + B_d.flatten() * u_opt

    # Store state and control input
    x_history.append(x.copy())
    u_history.append(u_opt)
    time.append(current_time + dt)

    # Observed dynamics for Bayesian inference
    observed_xdot = x[1]
    observed_ydot = x[3]
    observed_xdotdot = u_opt - mean_Cxx * observed_xdot - mean_Cxy * observed_ydot
    observed_ydotdot = -mean_Cyy * observed_ydot - mean_Cyx * observed_xdot

    # Update priors based on observed dynamics
    Cxx_likelihood = likelihood(observed_xdotdot, u_opt - Cxx_range * observed_xdot - Cxy_range * observed_ydot)
    Cxy_likelihood = likelihood(observed_xdotdot, u_opt - Cxx_range * observed_xdot - Cxy_range * observed_ydot)
    Cyy_likelihood = likelihood(observed_ydotdot, -Cyy_range * observed_ydot - Cyx_range * observed_xdot)
    Cyx_likelihood = likelihood(observed_ydotdot, -Cyy_range * observed_ydot - Cyx_range * observed_xdot)

    # Update posteriors by combining priors with likelihoods
    Cxx_posterior = Cxx_prior * Cxx_likelihood
    Cxy_posterior = Cxy_prior * Cxy_likelihood
    Cyy_posterior = Cyy_prior * Cyy_likelihood
    Cyx_posterior = Cyx_prior * Cyx_likelihood

    # Normalize posteriors (add epsilon to avoid division by zero)
    epsilon = 1e-8
    Cxx_prior = Cxx_posterior / (Cxx_posterior.sum() + epsilon)
    Cxy_prior = Cxy_posterior / (Cxy_posterior.sum() + epsilon)
    Cyy_prior = Cyy_posterior / (Cyy_posterior.sum() + epsilon)
    Cyx_prior = Cyx_posterior / (Cyx_posterior.sum() + epsilon)

    print(i, "Estimated C values:", mean_Cxx, mean_Cxy, mean_Cyy, mean_Cyx)

# Convert history to numpy arrays for plotting
x_history = np.array(x_history)
u_history = np.array(u_history)

# Plot the estimated C values over time
plt.figure(figsize=(10, 5))
plt.plot(time[:-1], mean_Cxx_values, label='Mean Estimate of $C_{xx}$', color='blue')
plt.plot(time[:-1], mean_Cxy_values, label='Mean Estimate of $C_{xy}$', color='cyan')
plt.plot(time[:-1], mean_Cyy_values, label='Mean Estimate of $C_{yy}$', color='green')
plt.plot(time[:-1], mean_Cyx_values, label='Mean Estimate of $C_{yx}$', color='lime')
plt.xlabel('Time (s)')
plt.ylabel('Estimated C Values')
plt.title('Estimated Damping Coefficients Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot the covariance trace over time
plt.figure(figsize=(10, 5))
plt.plot(time[:-1], cov_trace_values, label='Trace of Covariance', color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Trace of Covariance')
plt.title('Uncertainty in Parameter Estimates Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot the system response and the sinusoidal target
plt.figure(figsize=(10, 8))
for i in range(x_history.shape[1]):
    plt.plot(time, x_history[:, i], label=f'State x{i+1}')
plt.plot(time[:-1], target_y_values, color='orange', linestyle='--', label='Target y = sin(0.2 * sin(t * 0.1)) + 1')
plt.xlabel('Time (s)')
plt.ylabel('State Variables')
plt.title('State Response with Bayesian-Informed MPC (Sinusoidal Target)')
plt.legend()
plt.grid(True)
plt.show()

# Plot the control input over time
plt.figure(figsize=(10, 4))
plt.plot(time[:-1], u_history, label='Control Input u')
plt.xlabel('Time (s)')
plt.ylabel('Control Input')
plt.title('Optimal Control Input Over Time')
plt.legend()
plt.grid(True)
plt.show()
