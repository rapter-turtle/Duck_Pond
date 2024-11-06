import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Simulation parameters
time_step = 0.1  # Time step is 0.1s
total_time = 5  # 50 seconds total
time = np.arange(0, total_time, time_step)  # Time vector

# True values for s1 and s2 (unknown to the controller)
true_s1 = 1.0
true_s2 = 2.0

# Bayesian inference setup
# We'll search for the values of s1 and s2 over a grid
s1_range = np.linspace(-2.0, 3.0, 100)  # Possible values for s1
s2_range = np.linspace(-2.0, 3.0, 100)  # Possible values for s2

# Initial conditions
x_k = np.array([0.0, 0.0])  # Initial position [x1, x2]

# Define the likelihood function for angle observation (z_obs)
def likelihood_angle(s1, s2, z_obs, x1_obs, x2_obs, sigma=0.1):
    z_pred = np.arctan((x1_obs - s1) / (x2_obs - s2))
    return norm.pdf(z_obs, loc=z_pred, scale=sigma)

# Define the likelihood function for distance observation (z2_obs)
def likelihood_distance(s1, s2, z2_obs, x1_obs, x2_obs, sigma=0.1):
    z2_pred = np.sqrt((x1_obs - s1)**2 + (x2_obs - s2)**2)
    return norm.pdf(z2_obs, loc=z2_pred, scale=sigma)

# Define the distance function z2
def distance(x1, x2, s1, s2):
    return np.sqrt((x1 - s1)**2 + (x2 - s2)**2)

# Initialize posterior as a uniform distribution
posterior = np.ones((len(s1_range), len(s2_range)))

# Perform the Bayesian update over the time steps
for t in time:
    # Simulate x_k moving in a sine wave pattern
    x_k[0] = np.sin(0.1 * t)
    x_k[1] = np.sin(0.1 * t + np.pi / 4)

    # Observed angle (z_obs) based on the true s1 and s2
    z_obs = np.arctan((x_k[0] - true_s1) / (x_k[1] - true_s2))

    # Observed distance (z2_obs) between current position and true source
    z2_obs = distance(x_k[0], x_k[1], true_s1, true_s2)

    # Update the posterior with the new observations (both angle and distance)
    for i, s1 in enumerate(s1_range):
        for j, s2 in enumerate(s2_range):
            likelihood_angle_value = likelihood_angle(s1, s2, z_obs, x_k[0], x_k[1])
            likelihood_distance_value = likelihood_distance(s1, s2, z2_obs, x_k[0], x_k[1])
            posterior[i, j] *= likelihood_angle_value * likelihood_distance_value

    # Normalize the posterior after each step
    posterior /= np.sum(posterior)

    print(t)

# Plot the final posterior distribution
s1_grid, s2_grid = np.meshgrid(s1_range, s2_range, indexing='ij')
plt.figure(figsize=(8, 6))
plt.contourf(s1_grid, s2_grid, posterior, cmap='viridis')
plt.colorbar(label='Probability Density')
plt.scatter([true_s1], [true_s2], color='red', label='True Source', marker='x')
plt.title('Posterior Distribution of $s_1$ and $s_2$')
plt.xlabel('$s_1$')
plt.ylabel('$s_2$')
plt.legend()
plt.show()
