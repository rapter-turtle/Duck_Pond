import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the objective function to minimize (minimize x2)
def objective(x):
    return -x[2]  # Minimize x[2]

# Define the equality constraint
def constraint_eq(x):
    return x[0]*x[0] + x[1]*x[1] - x[2]*x[2]  # 2x0 + 2x1 = 2x2

# Define the initial guess
x0 = [0, 0, 0]

# Define the inequality constraints in bounds form
bounds = [(-5, 2), (-4, 3), (None, None)]  # -5 < x0 < 2, -4 < x1 < 3, no bounds on x2

# Define the constraints in dictionary form
constraints = [{'type': 'eq', 'fun': constraint_eq}]

# Perform the optimization using Sequential Least Squares Programming (SLSQP)
result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

# Print the results
print('Optimal variables:', result.x)
print('Optimal objective value (x2):', result.fun)
print('Optimization success:', result.success)
print('Optimization message:', result.message)

# Extract the optimal values
x_opt = result.x

# Define the range of values for x0 and x1
x0_range = np.linspace(-5, 5, 100)
x1_range = np.linspace(-5, 5, 100)

# Create a meshgrid for x0 and x1
X0, X1 = np.meshgrid(x0_range, x1_range)

# Calculate x2 based on the equation: x0^2 + x1^2 = x2^2
X2 = np.sqrt(X0**2 + X1**2)

# Create another surface for the negative root (since x2 can be positive or negative)
X2_neg = -np.sqrt(X0**2 + X1**2)

# Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the positive root surface
ax.plot_surface(X0, X1, X2, color='blue', alpha=0.6, edgecolor='none')

# Plot the negative root surface
ax.plot_surface(X0, X1, X2_neg, color='red', alpha=0.6, edgecolor='none')

# Set labels
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('x2')
ax.set_title('3D Surface of the Constraint x[0]^2 + x[1]^2 - x[2]^2 = 0')

# Show the plot
plt.show()