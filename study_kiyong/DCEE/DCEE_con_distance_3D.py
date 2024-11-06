import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

# Simulation parameters
time_step = 0.1  # Time step is 0.1s
total_time = 10  # Total simulation time in seconds
time = np.arange(0, total_time, time_step)  # Time vector

# True values for s1, s2, and s3 (unknown to the controller)
true_s1 = 4.0
true_s2 = 8.0
true_s3 = 5.0

# Bayesian inference setup
# We'll search for the values of s1, s2, and s3 over a grid
s1_range = np.linspace(0.0, 20.0, 50)  # Possible values for s1
s2_range = np.linspace(0.0, 20.0, 50)  # Possible values for s2
s3_range = np.linspace(0.0, 20.0, 50)  # Possible values for s3

# Initial conditions
x_k = np.array([0.0, 0.0, 0.0])  # Initial position [x1, x2, x3]

# Define the likelihood function for distance observation (z_obs) in 3D
def likelihood_distance(s1, s2, s3, z_obs, x1_obs, x2_obs, x3_obs, sigma=0.1):
    z_pred = np.sqrt((x1_obs - s1)**2 + (x2_obs - s2)**2 + (x3_obs - s3)**2)  # Predicted 3D distance
    return norm.pdf(z_obs, loc=z_pred, scale=sigma)

# Initialize posterior as a uniform distribution
posterior = np.ones((len(s1_range), len(s2_range), len(s3_range)))

# Lists to store mean and covariance at each time step
mean_s1_values = []
mean_s2_values = []
mean_s3_values = []
covariance_values = []
positions = [x_k.copy()]

# Perform the Bayesian update and control over time steps
for t in time:
    # Observed distance (z_obs) between the current position and the true source in 3D
    z_obs = np.sqrt((x_k[0] - true_s1)**2 + (x_k[1] - true_s2)**2 + (x_k[2] - true_s3)**2)

    # Update the posterior with the distance observation in 3D
    for i, s1 in enumerate(s1_range):
        for j, s2 in enumerate(s2_range):
            for k, s3 in enumerate(s3_range):
                likelihood_distance_value = likelihood_distance(s1, s2, s3, z_obs, x_k[0], x_k[1], x_k[2])
                posterior[i, j, k] *= likelihood_distance_value

    # Normalize the posterior after each step
    posterior /= np.sum(posterior)

    # Calculate the mean of s1, s2, and s3 using weighted average
    s1_grid, s2_grid, s3_grid = np.meshgrid(s1_range, s2_range, s3_range, indexing='ij')
    mean_s1 = np.sum(s1_grid * posterior)
    mean_s2 = np.sum(s2_grid * posterior)
    mean_s3 = np.sum(s3_grid * posterior)

    # Calculate the covariance of s1, s2, and s3
    cov_s1s1 = np.sum((s1_grid - mean_s1)**2 * posterior)
    cov_s2s2 = np.sum((s2_grid - mean_s2)**2 * posterior)
    cov_s3s3 = np.sum((s3_grid - mean_s3)**2 * posterior)
    cov_s1s2 = np.sum((s1_grid - mean_s1) * (s2_grid - mean_s2) * posterior)
    cov_s1s3 = np.sum((s1_grid - mean_s1) * (s3_grid - mean_s3) * posterior)
    cov_s2s3 = np.sum((s2_grid - mean_s2) * (s3_grid - mean_s3) * posterior)
    covariance_matrix = np.array([[cov_s1s1, cov_s1s2, cov_s1s3],
                                  [cov_s1s2, cov_s2s2, cov_s2s3],
                                  [cov_s1s3, cov_s2s3, cov_s3s3]])

    # Store mean and covariance values for plotting later
    mean_s1_values.append(mean_s1)
    mean_s2_values.append(mean_s2)
    mean_s3_values.append(mean_s3)
    covariance_values.append(covariance_matrix)

    # Define the cost function for optimizing control input
    def cost_function(u_k, x_k, mean_s1, mean_s2, mean_s3, covariance_matrix, time_step=0.1):
        # Predict the next position based on control input
        x_k1 = x_k + u_k * time_step
        # Cost includes distance to the estimated mean position and trace of covariance matrix
        distance_cost = np.linalg.norm(x_k1 - np.array([mean_s1, mean_s2, mean_s3]))**2
        trace_Q = np.trace(np.abs(covariance_matrix))  # Ensure trace is positive
        # trace_Q = np.trace(covariance_matrix)
        return distance_cost + trace_Q

    # Initial guess for the control input
    u0 = np.array([0.5, 0.0, 0.0])

    # Bounds for control inputs (for x1_dot, x2_dot, x3_dot)
    bounds = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]

    # Run optimization to find the best control input
    result = minimize(cost_function, u0, args=(x_k, mean_s1, mean_s2, mean_s3, covariance_matrix), method='SLSQP', bounds=bounds)
    u_k = result.x  # Optimal control input

    # Update position based on control input
    x_k = x_k + u_k * time_step
    positions.append(x_k.copy())

    print(f"Time: {t}, Position: {x_k}")

# Convert position list to a numpy array for easy plotting
positions = np.array(positions)

# Plot mean of s1, s2, and s3 over time
plt.figure(figsize=(10, 5))
plt.plot(time, mean_s1_values, label='Mean of $s_1$', color='blue')
plt.plot(time, mean_s2_values, label='Mean of $s_2$', color='green')
plt.plot(time, mean_s3_values, label='Mean of $s_3$', color='red')
plt.axhline(true_s1, color='blue', linestyle='--', label='True $s_1$')
plt.axhline(true_s2, color='green', linestyle='--', label='True $s_2$')
plt.axhline(true_s3, color='red', linestyle='--', label='True $s_3$')
plt.xlabel('Time (s)')
plt.ylabel('Mean Estimate')
plt.title('Mean Estimate of $s_1$, $s_2$, and $s_3$ Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot trajectory of the agent in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Agent Trajectory')
ax.scatter([true_s1], [true_s2], [true_s3], color='red', label='True Source', marker='x')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.set_title('Trajectory of the Agent and True Source Position')
ax.legend()
plt.show()
