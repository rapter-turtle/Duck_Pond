import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Simulation parameters
time_step = 0.1  # Time step is 0.1s
total_time = 25  # 20 seconds total
time = np.arange(0, total_time, time_step)  # Time vector

# True values for s1 and s2 (unknown to the controller)
true_s1 = 4.0
true_s2 = 8.0

# Bayesian inference setup
# We'll search for the values of s1 and s2 over a grid
s1_range = np.linspace(-5.0, 20.0, 100)  # Possible values for s1
s2_range = np.linspace(-5.0, 20.0, 100)  # Possible values for s2

# Initial conditions
x_k = np.array([0.0, 0.0])  # Initial position [x1, x2]
previous_error = np.array([0.0, 0.0])  # Initialize previous error for the derivative term

# PD Controller gains
Kp = 1.0  # Proportional gain
Kd = 0.1  # Derivative gain
control_limit = 0.5  # Limit for control input (max speed set to ±0.5)

# Define the likelihood function for distance observation (z_obs)
def likelihood_distance(s1, s2, z_obs, x1_obs, x2_obs, sigma=0.1):
    z_pred = np.sqrt((x1_obs - s1)**2 + (x2_obs - s2)**2)  # Predicted distance
    return norm.pdf(z_obs, loc=z_pred, scale=sigma)

# Initialize posterior as a uniform distribution
posterior = np.ones((len(s1_range), len(s2_range)))

# Lists to store mean and covariance at each time step
mean_s1_values = []
mean_s2_values = []
covariance_values = []
positions = [x_k.copy()]

# Perform the Bayesian update and PD control over time steps
for t in time:
    # Observed distance (z_obs) between the current position and the true source
    z_obs = np.sqrt((x_k[0] - true_s1)**2 + (x_k[1] - true_s2)**2)

    # Update the posterior with the distance observation only
    for i, s1 in enumerate(s1_range):
        for j, s2 in enumerate(s2_range):
            likelihood_distance_value = likelihood_distance(s1, s2, z_obs, x_k[0], x_k[1])
            posterior[i, j] *= likelihood_distance_value

    # Normalize the posterior after each step
    posterior /= np.sum(posterior)

    # Calculate the mean of s1 and s2 using weighted average
    s1_grid, s2_grid = np.meshgrid(s1_range, s2_range, indexing='ij')
    mean_s1 = np.sum(s1_grid * posterior)
    mean_s2 = np.sum(s2_grid * posterior)

    # Calculate the covariance of s1 and s2
    cov_s1s1 = np.sum((s1_grid - mean_s1)**2 * posterior)
    cov_s2s2 = np.sum((s2_grid - mean_s2)**2 * posterior)
    cov_s1s2 = np.sum((s1_grid - mean_s1) * (s2_grid - mean_s2) * posterior)
    covariance_matrix = np.array([[cov_s1s1, cov_s1s2], [cov_s1s2, cov_s2s2]])

    # Store mean and covariance values for plotting later
    mean_s1_values.append(mean_s1)
    mean_s2_values.append(mean_s2)
    covariance_values.append(covariance_matrix)

    # Calculate the PD control input
    target_position = np.array([mean_s1, mean_s2])
    error = target_position - x_k
    derivative = (error - previous_error) / time_step
    control_input = Kp * error + Kd * derivative

    # Limit the control input to the range [-0.5, 0.5]
    control_input = np.clip(control_input, -control_limit, control_limit)

    # Update position based on PD control input
    x_k = x_k + control_input * time_step
    positions.append(x_k.copy())

    # Update previous error for the next iteration
    previous_error = error

    print(f"Time: {t}, Position: {x_k}, Control Input: {control_input}")

# Convert position list to a numpy array for easy plotting
positions = np.array(positions)

# Plot the final posterior distribution
plt.figure(figsize=(8, 6))
plt.contourf(s1_grid, s2_grid, posterior, cmap='viridis')
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

# Plot covariance of s1 and s2 over time
cov_s1s1_values = [cov[0, 0] for cov in covariance_values]
cov_s2s2_values = [cov[1, 1] for cov in covariance_values]
cov_s1s2_values = [cov[0, 1] for cov in covariance_values]

plt.figure(figsize=(10, 5))
plt.plot(time, cov_s1s1_values, label='Variance of $s_1$', color='blue')
plt.plot(time, cov_s2s2_values, label='Variance of $s_2$', color='green')
plt.plot(time, cov_s1s2_values, label='Covariance of $s_1$ and $s_2$', color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Covariance')
plt.title('Covariance of $s_1$ and $s_2$ Over Time')
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
