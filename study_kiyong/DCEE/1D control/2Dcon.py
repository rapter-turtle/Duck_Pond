import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Simulation parameters
time_step = 0.1  # Time step
total_time = 20  # Total simulation time in seconds
time = np.arange(0, total_time, time_step)  # Time vector

# True values for Cx1, Cx2, Cy1, and Cy2 (unknown to the controller)
true_Cx1, true_Cx2 = 0.5, 0.2  # Damping coefficients for x dynamics
true_Cy1, true_Cy2 = 0.3, 0.4  # Damping coefficients for y dynamics

# Initial conditions for state [x, xdot, y, ydot]
state = np.array([0.0, 0.0, 0.0, 0.0])  # Start at (x, y) = (0, 0), velocities (xdot, ydot) = (0, 0)

# Lists to store estimates, states, and costs over time
mean_Cx1_values, mean_Cx2_values = [], []
mean_Cy1_values, mean_Cy2_values = [], []
x_values, xdot_values, y_values, ydot_values = [], [], [], []
distance_cost_values, uncertainty_cost_values, total_cost_values = [], [], []
trajectory_x_values, trajectory_y_values = [], []

# GP regression models with initial kernels
kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))  # Constant * RBF kernel
gp_Cx1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
gp_Cx2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
gp_Cy1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
gp_Cy2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)

# Observations list for training
X_observed, y_Cx1, y_Cx2, y_Cy1, y_Cy2 = [], [], [], [], []

# Constraint functions to ensure xdot and ydot remain within [-1, 1]
def xdot_constraint(controls, state, mean_Cx1, mean_Cx2, time_step=0.1):
    ux, _ = controls
    xdot = state[1]
    ydot = state[3]
    xdot_next = xdot + (ux - mean_Cx1 * xdot - mean_Cx2 * ydot) * time_step
    return 1 - np.abs(xdot_next)  # Ensures -1 <= xdot_next <= 1

def ydot_constraint(controls, state, mean_Cy1, mean_Cy2, time_step=0.1):
    _, uy = controls
    xdot = state[1]
    ydot = state[3]
    ydot_next = ydot + (uy - mean_Cy1 * xdot - mean_Cy2 * ydot) * time_step
    return 1 - np.abs(ydot_next)  # Ensures -1 <= ydot_next <= 1

# Perform the Gaussian Process regression and control over time steps
for t in time:
    # Target positions for x and y based on the trajectory
    target_x = 3 * np.sin(0.1 * t) + 1
    target_y = 3 * np.cos(0.1 * t) + 1
    trajectory_x_values.append(target_x)
    trajectory_y_values.append(target_y)

    # Observed values based on true dynamics with true Cx1, Cx2, Cy1, Cy2
    observed_x, observed_xdot = state[0], state[1]
    observed_y, observed_ydot = state[2], state[3]
    observed_xdotdot = -true_Cx1 * observed_xdot - true_Cx2 * observed_ydot
    observed_ydotdot = -true_Cy1 * observed_xdot - true_Cy2 * observed_ydot

    # Record the observed input data (velocities) and corresponding outputs
    X_observed.append([observed_xdot, observed_ydot])
    y_Cx1.append(observed_xdotdot / observed_xdot if observed_xdot != 0 else 0)  # avoid divide by zero
    y_Cx2.append(observed_xdotdot / observed_ydot if observed_ydot != 0 else 0)
    y_Cy1.append(observed_ydotdot / observed_xdot if observed_xdot != 0 else 0)
    y_Cy2.append(observed_ydotdot / observed_ydot if observed_ydot != 0 else 0)

    # Train GP regression models if enough data points are available
    if len(X_observed) > 5:
        gp_Cx1.fit(X_observed, y_Cx1)
        gp_Cx2.fit(X_observed, y_Cx2)
        gp_Cy1.fit(X_observed, y_Cy1)
        gp_Cy2.fit(X_observed, y_Cy2)

        # Predict damping coefficients and their uncertainties
        # mean_Cx1, std_Cx1 = gp_Cx1.predict([[observed_xdot, observed_ydot]], return_std=True)
        # mean_Cx2, std_Cx2 = gp_Cx2.predict([[observed_xdot, observed_ydot]], return_std=True)
        # mean_Cy1, std_Cy1 = gp_Cy1.predict([[observed_xdot, observed_ydot]], return_std=True)
        # mean_Cy2, std_Cy2 = gp_Cy2.predict([[observed_xdot, observed_ydot]], return_std=True)

        mean_Cx1, std_Cx1 = gp_Cx1.predict([[observed_xdot, observed_ydot]], return_std=True)
        mean_Cx2, std_Cx2 = gp_Cx2.predict([[observed_xdot, observed_ydot]], return_std=True)
        mean_Cy1, std_Cy1 = gp_Cy1.predict([[observed_xdot, observed_ydot]], return_std=True)
        mean_Cy2, std_Cy2 = gp_Cy2.predict([[observed_xdot, observed_ydot]], return_std=True)

        # Convert predictions to scalars for appending to lists
        mean_Cx1, std_Cx1 = mean_Cx1.item(), std_Cx1.item()
        mean_Cx2, std_Cx2 = mean_Cx2.item(), std_Cx2.item()
        mean_Cy1, std_Cy1 = mean_Cy1.item(), std_Cy1.item()
        mean_Cy2, std_Cy2 = mean_Cy2.item(), std_Cy2.item()

        # Calculate uncertainty cost as the sum of variances
        uncertainty_cost = std_Cx1**2 + std_Cx2**2 + std_Cy1**2 + std_Cy2**2
    else:
        mean_Cx1, mean_Cx2, mean_Cy1, mean_Cy2 = 0.5, 0.2, 0.3, 0.4  # Initial estimates
        uncertainty_cost = 0  # No uncertainty cost if not enough data

    mean_Cx1_values.append(mean_Cx1)
    mean_Cx2_values.append(mean_Cx2)
    mean_Cy1_values.append(mean_Cy1)
    mean_Cy2_values.append(mean_Cy2)
    uncertainty_cost_values.append(uncertainty_cost)

    # Cost function for control in x and y directions
    def cost_function(controls, state, mean_Cx1, mean_Cx2, mean_Cy1, mean_Cy2, target_x, target_y, uncertainty_cost, time_step=0.1):
        ux, uy = controls
        x, xdot, y, ydot = state
        xdotdot = ux - mean_Cx1 * xdot - mean_Cx2 * ydot
        ydotdot = uy - mean_Cy1 * xdot - mean_Cy2 * ydot
        x_next = x + xdot * time_step + 0.5 * xdotdot * time_step**2
        xdot_next = xdot + xdotdot * time_step
        y_next = y + ydot * time_step + 0.5 * ydotdot * time_step**2
        ydot_next = ydot + ydotdot * time_step

        # Define costs
        distance_cost = (x_next - target_x) ** 2 + (y_next - target_y) ** 2
        total_cost = distance_cost + 100000 * uncertainty_cost  # Add uncertainty cost
        return total_cost

    # Initial guess for control inputs
    u0 = [0.0, 0.0]  # Initial control guesses for ux and uy

    # Define constraints to keep xdot and ydot within [-1, 1]
    constraints = [
        {'type': 'ineq', 'fun': xdot_constraint, 'args': (state, mean_Cx1, mean_Cx2, time_step)},
        {'type': 'ineq', 'fun': ydot_constraint, 'args': (state, mean_Cy1, mean_Cy2, time_step)}
    ]

    # Run optimization to find the best control inputs with velocity constraints
    result = minimize(
        cost_function, u0, 
        args=(state, mean_Cx1, mean_Cx2, mean_Cy1, mean_Cy2, target_x, target_y, uncertainty_cost),
        bounds=[(-10, 10), (-10, 10)],  # Control input limits for u_x and u_y
        constraints=constraints,
        method='SLSQP'
    )
    u_x, u_y = result.x  # Optimal control inputs

    # Apply control inputs to update state
    xdotdot = u_x - mean_Cx1 * state[1] - mean_Cx2 * state[3]
    ydotdot = u_y - mean_Cy1 * state[1] - mean_Cy2 * state[3]
    state[0] = state[0] + state[1] * time_step + 0.5 * xdotdot * time_step**2
    state[1] = state[1] + xdotdot * time_step
    state[2] = state[2] + state[3] * time_step + 0.5 * ydotdot * time_step**2
    state[3] = state[3] + ydotdot * time_step

    # Store state and costs
    x_values.append(state[0])
    xdot_values.append(state[1])
    y_values.append(state[2])
    ydot_values.append(state[3])
    distance_cost = (state[0] - target_x) ** 2 + (state[2] - target_y) ** 2
    distance_cost_values.append(distance_cost)
    total_cost_values.append(distance_cost + uncertainty_cost)

    print("position : ",state)


# Plot the estimated Cx1, Cx2, Cy1, and Cy2 over time
plt.figure(figsize=(10, 5))
plt.plot(time[:len(mean_Cx1_values)], mean_Cx1_values, label='Mean Estimate of $Cx1$', color='blue')
plt.plot(time[:len(mean_Cx2_values)], mean_Cx2_values, label='Mean Estimate of $Cx2$', color='cyan')
plt.plot(time[:len(mean_Cy1_values)], mean_Cy1_values, label='Mean Estimate of $Cy1$', color='green')
plt.plot(time[:len(mean_Cy2_values)], mean_Cy2_values, label='Mean Estimate of $Cy2$', color='lime')
plt.axhline(true_Cx1, color='blue', linestyle='--', label='True $Cx1$')
plt.axhline(true_Cx2, color='cyan', linestyle='--', label='True $Cx2$')
plt.axhline(true_Cy1, color='green', linestyle='--', label='True $Cy1$')
plt.axhline(true_Cy2, color='lime', linestyle='--', label='True $Cy2$')
plt.xlabel('Time (s)')
plt.ylabel('Estimates of $Cx1$, $Cx2$, $Cy1$, $Cy2$')
plt.title('Estimates of $Cx1$, $Cx2$, $Cy1$, and $Cy2$ Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot the x, y positions and their target trajectories over time
plt.figure(figsize=(10, 5))
plt.plot(time[:len(x_values)], x_values, label='$x$', color='blue')
plt.plot(time[:len(y_values)], y_values, label='$y$', color='green')
plt.plot(time[:len(trajectory_x_values)], trajectory_x_values, label='Target Trajectory $x$', color='red', linestyle='--')
plt.plot(time[:len(trajectory_y_values)], trajectory_y_values, label='Target Trajectory $y$', color='orange', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.title('Position Variables Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot the state variables xdot and ydot over time
plt.figure(figsize=(10, 5))
plt.plot(time[:len(xdot_values)], xdot_values, label='$\dot{x}$', color='blue')
plt.plot(time[:len(ydot_values)], ydot_values, label='$\dot{y}$', color='green')
plt.axhline(1, color='red', linestyle='--', label='Velocity Limit')
plt.axhline(-1, color='red', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.title('Velocity Variables Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot cost components over time
plt.figure(figsize=(10, 5))
plt.plot(time[:len(distance_cost_values)], distance_cost_values, label='Distance Cost', color='blue')
plt.plot(time[:len(uncertainty_cost_values)], uncertainty_cost_values, label='Uncertainty Cost', color='green')
plt.plot(time[:len(total_cost_values)], total_cost_values, label='Total Cost', color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Cost')
plt.title('Cost Components Over Time')
plt.legend()
plt.grid(True)
plt.show()
