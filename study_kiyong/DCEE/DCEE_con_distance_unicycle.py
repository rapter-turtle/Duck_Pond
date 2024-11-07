import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Simulation parameters
time_step = 0.1  # Time step is 0.1s
total_time = 50  # Total simulation time in seconds
time = np.arange(0, total_time, time_step)  # Time vector

# True values for s1 and s2 (unknown to the controller)
true_s1 = 19.0
true_s2 = 5.0

# Bayesian inference setup
s1_range = torch.linspace(-5.0, 20.0, 1000, device=device)  # Dense grid for s1
s2_range = torch.linspace(-5.0, 20.0, 1000, device=device)  # Dense grid for s2

# Create grids for s1 and s2 for vectorized operations
s1_grid, s2_grid = torch.meshgrid(s1_range, s2_range, indexing='ij')
posterior = torch.ones_like(s1_grid, device=device)  # Initialize posterior

# Initial conditions
x_k = torch.tensor([0.0, 0.0], device=device)  # Initial position [x, y]
theta_k = torch.tensor(3.141592, device=device)  # Initial orientation

# Define control limits
v_limit = 0.5    # Maximum linear velocity
omega_limit = 1.0 # Maximum angular velocity

# Lists to store mean and cost values over time
mean_s1_values = []
mean_s2_values = []
positions = [x_k.cpu().numpy()]
distance_cost_values = []
trace_Q_values = []
total_cost_values = []
loop_times = []  # Store calculation time for each loop

# Define the likelihood function for distance observation (z_obs)
def likelihood_distance(s1, s2, z_obs, x1_obs, x2_obs, sigma=0.1):
    z_pred = torch.sqrt((x1_obs - s1)**2 + (x2_obs - s2)**2)  # Predicted distance
    return torch.exp(-0.5 * ((z_obs - z_pred) / sigma) ** 2) / (sigma * torch.sqrt(torch.tensor(2 * np.pi)))

# Perform the Bayesian update and control over time steps
for t in time:
    # Start time measurement
    start_event = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    end_event = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    if device == 'cuda':
        start_event.record()
    else:
        start_time = time.time()

    # Observed distance (z_obs) between the current position and the true source
    z_obs = torch.sqrt((x_k[0] - true_s1)**2 + (x_k[1] - true_s2)**2).to(device)

    # Update the posterior with the distance observation only
    z_pred = torch.sqrt((x_k[0] - s1_grid)**2 + (x_k[1] - s2_grid)**2)
    likelihood = likelihood_distance(s1_grid, s2_grid, z_obs, x_k[0], x_k[1])
    posterior *= likelihood
    posterior /= posterior.sum()  # Normalize posterior

    # Calculate the mean of s1 and s2 using weighted average
    mean_s1 = (s1_grid * posterior).sum().item()
    mean_s2 = (s2_grid * posterior).sum().item()

    # Calculate the covariance of s1 and s2
    cov_s1s1 = ((s1_grid - mean_s1)**2 * posterior).sum().item()
    cov_s2s2 = ((s2_grid - mean_s2)**2 * posterior).sum().item()
    cov_s1s2 = ((s1_grid - mean_s1) * (s2_grid - mean_s2) * posterior).sum().item()
    covariance_matrix = np.array([[cov_s1s1, cov_s1s2], [cov_s1s2, cov_s2s2]])

    # Store mean values for plotting later
    mean_s1_values.append(mean_s1)
    mean_s2_values.append(mean_s2)

    # Convert Torch tensors to numpy arrays for the cost function
    mean_position = np.array([mean_s1, mean_s2])
    x_k_np = x_k.cpu().numpy()
    theta_k_np = theta_k.cpu().item()

    # Define the cost function for optimizing control input
    def cost_function(u_k, x_k_np, theta_k_np, mean_position, covariance_matrix, time_step=0.1):
        v, omega = u_k
        theta_k1 = theta_k_np + omega * time_step
        x_k1 = x_k_np + np.array([v * np.cos(theta_k1), v * np.sin(theta_k1)]) * time_step

        # Calculate the individual costs
        distance_cost = np.linalg.norm(x_k1 - mean_position)**2
        trace_Q = np.trace(covariance_matrix)  # Total uncertainty measure
        total_cost = distance_cost + 100000*trace_Q #weight is important

        # Return total cost for optimization
        return total_cost

    # Initial guess for the control input [v, omega]
    u0 = np.array([0.0, 0.0])

    # Bounds for control inputs (linear velocity v and angular velocity omega)
    bounds = [(-0.2, v_limit), (-omega_limit, omega_limit)]  # v is bounded between 0 and v_limit, omega between -omega_limit and omega_limit

    # Run optimization to find the best control input
    result = minimize(cost_function, u0, args=(x_k_np, theta_k_np, mean_position, covariance_matrix), method='SLSQP', bounds=bounds)
    v, omega = result.x  # Optimal control inputs: linear and angular velocities

    # Calculate the costs for plotting
    distance_cost = np.linalg.norm(x_k_np + np.array([v * np.cos(theta_k_np + omega * time_step), v * np.sin(theta_k_np + omega * time_step)]) * time_step - mean_position)**2
    trace_Q = np.trace(covariance_matrix)
    total_cost = distance_cost + trace_Q
    distance_cost_values.append(distance_cost)
    trace_Q_values.append(trace_Q)
    total_cost_values.append(total_cost)

    # Update position and orientation based on control input
    theta_k = theta_k + omega * time_step
    x_k = x_k + torch.tensor([v * torch.cos(theta_k), v * torch.sin(theta_k)], device=device) * time_step
    positions.append(x_k.cpu().numpy())

    # End time measurement
    if device == 'cuda':
        end_event.record()
        torch.cuda.synchronize()
        loop_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
    else:
        loop_time = time.time() - start_time

    loop_times.append(loop_time)
    print(f"Time: {t:.1f}, Position: {x_k.cpu().numpy()}, Orientation: {theta_k.cpu().item():.2f}, Control Input: v={v:.2f}, omega={omega:.2f}, Loop Time: {loop_time:.4f} s")

# Convert position list to a numpy array for easy plotting
positions = np.array(positions)

# Plot the final posterior distribution
plt.figure(figsize=(8, 6))
plt.contourf(s1_range.cpu().numpy(), s2_range.cpu().numpy(), posterior.cpu().numpy(), cmap='viridis')
plt.colorbar(label='Probability Density')
plt.scatter([true_s1], [true_s2], color='red', label='True Source', marker='x')
plt.title('Final Posterior Distribution of $s_1$ and $s_2$')
plt.xlabel('$s_1$')
plt.ylabel('$s_2$')
plt.legend()
plt.show()

# Plot mean of s1 and s2 over time
plt.figure(figsize=(10, 5))
plt.plot(time, mean_s1_values, label='Mean of $s_1$', color='blue')
plt.plot(time, mean_s2_values, label='Mean of $s_2$', color='green')
plt.axhline(true_s1, color='blue', linestyle='--', label='True $s_1$')
plt.axhline(true_s2, color='green', linestyle='--', label='True $s_2$')
plt.xlabel('Time (s)')
plt.ylabel('Mean Estimate')
plt.title('Mean Estimate of $s_1$ and $s_2$ Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot cost values over time
plt.figure(figsize=(10, 5))
plt.plot(time, distance_cost_values, label='Distance Cost', color='blue')
plt.plot(time, trace_Q_values, label='Trace of Covariance', color='green')
plt.plot(time, total_cost_values, label='Total Cost (Distance + Trace)', color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Cost')
plt.title('Cost Components Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot the trajectory of the agent
plt.figure(figsize=(8, 6))
plt.plot(positions[:, 0], positions[:, 1], label='Agent Trajectory')
plt.scatter([true_s1], [true_s2], color='red', label='True Source', marker='x')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Trajectory of the Agent and True Source Position')
plt.legend()
plt.grid(True)
plt.show()
