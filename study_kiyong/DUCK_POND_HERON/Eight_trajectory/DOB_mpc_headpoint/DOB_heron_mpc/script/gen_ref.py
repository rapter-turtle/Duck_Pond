import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

def generate_figure_eight_trajectory(tfinal, dt, translation, theta, num_loops=3, start_point=(0, 0), amplitude=100, wavelength=100, velocity=1.0):
    t = np.arange(0, tfinal, dt)
    
    # Generate a figure-eight trajectory using the given parametric equations
    x = np.linspace(0, num_loops * 2 * np.pi, int(tfinal / dt))
    y = amplitude * np.sin(x) * np.cos(x) + start_point[1]
    
    # Adjust for the start point
    x = amplitude * np.sin(x) + start_point[0]
    
    # Calculate the arc length for each point
    dx = np.gradient(x)
    dy = np.gradient(y)
    ds = np.sqrt(dx**2 + dy**2)
    s = cumtrapz(ds, initial=0)
    
    # Determine the desired uniform arc length spacing
    total_distance = s[-1]
    num_points = int(total_distance / (velocity * dt))
    s_uniform = np.linspace(0, total_distance, num_points)
    
    # Interpolate the x and y positions based on uniform arc length
    x_uniform = interp1d(s, x)(s_uniform)
    y_uniform = interp1d(s, y)(s_uniform)
    
    # Recalculate derivatives after interpolation
    dx_uniform = np.gradient(x_uniform, dt)
    dy_uniform = np.gradient(y_uniform, dt)
    velocity_magnitudes = np.sqrt(dx_uniform**2 + dy_uniform**2)
    
    # Compute headings and rotational speeds
    headings = np.arctan2(dy_uniform, dx_uniform)
    headings = np.unwrap(headings)
    rot_speed = np.gradient(headings, dt)
    
    # Create the reference trajectory
    positions = np.vstack((x_uniform, y_uniform)).T
    ref = np.hstack((positions, headings.reshape(-1, 1), velocity_magnitudes.reshape(-1, 1), rot_speed.reshape(-1, 1)))
    
    # Apply translation and rotation transformations
    ref = transform_trajectory(ref, translation, theta)
    
    # Save the reference trajectory data
    np.save('ref_data.npy', ref)

    return ref

def transform_trajectory(trajectory, translation, theta):
    # Translation and rotation
    translated_positions = trajectory[:, :2] + np.array(translation)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])
    rotated_positions = translated_positions @ rotation_matrix.T

    # Adjust headings by adding rotation angle
    adjusted_headings = trajectory[:, 2] + theta

    # Maintain velocity magnitudes and rotational speeds
    velocity_magnitudes = trajectory[:, 3]
    rot_speed = trajectory[:, 4]

    transformed_trajectory = np.hstack((rotated_positions,
                                        adjusted_headings.reshape(-1, 1),
                                        velocity_magnitudes.reshape(-1, 1),
                                        rot_speed.reshape(-1, 1)))

    return transformed_trajectory

if __name__ == '__main__':
    tfinal = 10  # Duration of the trajectory
    dt = 0.01  # Time step
    translation = (0, 0)  # No translation
    theta = np.pi / 4  # Rotation angle in radians (e.g., pi/4 for 45 degrees)
    num_loops = 1  # Number of figure-eight loops
    start_point = (0, 0)  # Starting point of the trajectory
    amplitude = 100  # Amplitude of the figure-eight
    wavelength = 100  # Not used in this context, adjust if necessary
    velocity = 1.0  # Desired constant velocity

    positions = generate_figure_eight_trajectory(tfinal, dt, translation, theta, num_loops, start_point, amplitude, wavelength, velocity)

    # Plot the generated figure-eight trajectory
    plt.plot(positions[:, 0], positions[:, 1], 'b-')
    plt.title("Figure-Eight Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.axis('equal')
    plt.show()
