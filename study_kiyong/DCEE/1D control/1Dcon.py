import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import torch
import time

# Simulation parameters
time_step = 0.1  # Time step
total_time = 50  # Total simulation time in seconds
time_steps = np.arange(0, total_time, time_step)  # Time vector

# True value for C (unknown to the controller)
true_C = 0.5  # Damping coefficient

# Initial conditions for state [x, xdot]
state = np.array([0.0, 0.0])  # Start at x=0, xdot=0

# Lists to store estimates and state over time
damping_term_values = []
x_values = []
xdot_values = []
distance_cost_values = []
uncertainty_cost_values = []
total_cost_values = []
trajectory_values = []

# Gaussian Process class definition
class GaussianProcess(torch.nn.Module):
    def __init__(self, kernel):
        super(GaussianProcess, self).__init__()
        self.kernel = kernel
    
    def forward(self, X_train, y_train, X_test, noise=1e-5):
        # Compute the kernel matrices
        K = self.kernel(X_train, X_train) + noise * torch.eye(len(X_train))
        K_s = self.kernel(X_train, X_test)
        K_ss = self.kernel(X_test, X_test) + noise * torch.eye(len(X_test))
        
        # Compute the mean and covariance of the posterior distribution
        K_inv = torch.linalg.inv(K)
        mu = K_s.T @ K_inv @ y_train
        cov = K_ss - K_s.T @ K_inv @ K_s
        return mu, cov

# Kernel function definition
class RBFKernel(torch.nn.Module):
    def __init__(self, length_scale=1.0):
        super(RBFKernel, self).__init__()
        self.length_scale = torch.nn.Parameter(torch.tensor(length_scale))
    
    def forward(self, X1, X2):
        sqdist = torch.cdist(X1 / self.length_scale, X2 / self.length_scale) ** 2
        return torch.exp(-0.5 * sqdist)

# Initialize GP model and kernel
kernel = RBFKernel(length_scale=1.0)
gp = GaussianProcess(kernel)

# Observations list for training the GP (limited to 10 most diverse points)
X_observed = []  # Observed xdot values
y_observed = []  # Observed values of the damping function term

# Threshold to control data diversity
variation_threshold = 0.5  # Minimum xdot difference required to keep a new point

# To track the computation time of each loop iteration
computation_times = []

# Perform the Gaussian Process regression and control over time steps
for t in time_steps:
    # Start timing the iteration
    start_time = time.time()

    # Target position for x based on the trajectory
    target_x = 3 * np.sin(0.3 * t) + 1
    trajectory_values.append(target_x)

    # Observed values based on the true dynamics
    observed_x = state[0]  # Observed position
    observed_xdot = state[1]  # Observed velocity
    observed_damping_term = -true_C * np.exp(-np.abs(observed_xdot)) * observed_xdot  # True unknown damping function

    # Add data point only if it's diverse enough from existing points in X_observed
    if len(X_observed) < 10 or all(abs(observed_xdot - x[0]) > variation_threshold for x in X_observed):
        X_observed.append([observed_xdot])
        y_observed.append(observed_damping_term)

    # Limit to the most recent 10 diverse data points
    if len(X_observed) > 10:
        X_observed.pop(0)
        y_observed.pop(0)

    # Train GP regression model if enough data points are available
    if len(X_observed) == 10:
        X_train = torch.tensor(X_observed, dtype=torch.float32)
        y_train = torch.tensor(y_observed, dtype=torch.float32)
        mean_damping_term, cov_damping_term = gp(X_train, y_train, X_train)
        std_damping_term = torch.sqrt(torch.diag(cov_damping_term)).detach().numpy()
        mean_damping_term = mean_damping_term[-1].item()
        uncertainty_cost = std_damping_term[-1] ** 2
    else:
        mean_damping_term = -0.5 * np.exp(-observed_xdot) * observed_xdot  # Initial estimate
        uncertainty_cost = 0  # No uncertainty cost if not enough data

    damping_term_values.append(mean_damping_term)
    uncertainty_cost_values.append(uncertainty_cost)

    # Cost function for control
    def cost_function(u, state, mean_damping_term, target_x, uncertainty_cost, time_step=0.1):
        x, xdot = state
        xdotdot = u + mean_damping_term  # mean_damping_term approximates -C*exp(-xdot)*xdot
        x_next = x + xdot * time_step + 0.5 * xdotdot * time_step**2
        xdot_next = xdot + xdotdot * time_step

        # Define costs
        distance_cost = (x_next - target_x) ** 2
        total_cost = distance_cost #+ 1000000 * uncertainty_cost  # Include uncertainty cost
        return total_cost

    # Dynamic velocity limit
    velocity_limit = 1

    # Constraint to ensure xdot_next < velocity_limit
    def velocity_constraint(u, state, mean_damping_term, velocity_limit, time_step=0.1):
        xdot = state[1]
        xdot_next = xdot + (u + mean_damping_term) * time_step  # Use mean_damping_term in place of the true unknown function
        return velocity_limit - xdot_next  # Ensures xdot_next < velocity_limit

    # Initial guess for control input
    u0 = [0.0]

    # Run optimization to find best control input with constraints
    constraints = {'type': 'ineq', 'fun': velocity_constraint, 'args': (state, mean_damping_term, velocity_limit, time_step)}
    result = minimize(cost_function, u0, args=(state, mean_damping_term, target_x, uncertainty_cost), 
                      bounds=[(-10, 10)], method='SLSQP', constraints=constraints)
    u_optimal = result.x[0]

    # Apply control input to update state
    xdotdot = u_optimal + mean_damping_term
    state[0] = state[0] + state[1] * time_step + 0.5 * xdotdot * time_step**2
    state[1] = state[1] + xdotdot * time_step

    # Store state and costs
    x_values.append(state[0])
    xdot_values.append(state[1])
    distance_cost = (state[0] - target_x) ** 2
    distance_cost_values.append(distance_cost)
    total_cost_values.append(distance_cost + uncertainty_cost)

    # End timing the iteration and store the computation time
    end_time = time.time()
    computation_times.append(end_time - start_time)

# Calculate the average and maximum computation times
average_computation_time = np.mean(computation_times)
max_computation_time = np.max(computation_times)

print(f"Average computation time per iteration: {average_computation_time:.6f} seconds")
print(f"Maximum computation time per iteration: {max_computation_time:.6f} seconds")

# Plot the estimated damping term over time
plt.figure(figsize=(10, 5))
plt.plot(time_steps, damping_term_values, label='Mean Estimate of Damping Term', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Estimate of Damping Term')
plt.title('Estimate of Damping Term Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot the state variables x and xdot over time
plt.figure(figsize=(10, 5))
plt.plot(time_steps, x_values, label='$x$', color='blue')
plt.plot(time_steps, xdot_values, label='$\dot{x}$', color='green')
plt.plot(time_steps, trajectory_values, label='Target Trajectory $x=3 \sin(0.3t) + 5$', color='red', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.title('State Variables Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot cost components over time
plt.figure(figsize=(10, 5))
plt.plot(time_steps, distance_cost_values, label='Distance Cost', color='blue')
plt.plot(time_steps, uncertainty_cost_values, label='Uncertainty Cost from GP', color='green')
plt.plot(time_steps, total_cost_values, label='Total Cost', color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Cost')
plt.title('Cost Components Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot the Gaussian Process model of the damping term vs. xdot
xdot_test = np.linspace(-2, 2, 100).reshape(-1, 1)  # Range of xdot for GP prediction
X_test = torch.tensor(xdot_test, dtype=torch.float32)

if len(X_observed) == 10:  # Ensure GP is trained
    mean_damping_term_test, cov_damping_term_test = gp(X_train, y_train, X_test)
    std_damping_term_test = torch.sqrt(torch.diag(cov_damping_term_test)).detach().numpy()
    mean_damping_term_test = mean_damping_term_test.detach().numpy()
else:
    mean_damping_term_test = np.zeros(len(xdot_test))
    std_damping_term_test = np.zeros(len(xdot_test))

plt.figure(figsize=(10, 5))
plt.plot(xdot_test, mean_damping_term_test, label='GP Mean Damping Term', color='blue')
plt.fill_between(
    xdot_test.flatten(),
    mean_damping_term_test - 2 * std_damping_term_test,
    mean_damping_term_test + 2 * std_damping_term_test,
    color='lightblue', alpha=0.5, label='Confidence Interval (±2σ)'
)
plt.xlabel('$\dot{x}$')
plt.ylabel('Damping Term')
plt.title('Gaussian Process Model of Damping Term vs. $\dot{x}$')
plt.legend()
plt.grid(True)
plt.show()
