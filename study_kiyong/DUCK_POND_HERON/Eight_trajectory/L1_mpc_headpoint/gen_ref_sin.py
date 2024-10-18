import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

def generate_snake_s_shape_trajectory(tfinal, dt, translation, theta, num_s_shapes=3, start_point=(0, 0), amplitude=5, wavelength=10, velocity=1.0):
    t = np.arange(0, tfinal, dt)
    
    # Generate a snake-like S-shape using sine function
    x = np.linspace(0, num_s_shapes * wavelength, int(tfinal / dt))
    y = amplitude * np.sin(2 * np.pi * x / wavelength) + start_point[1]
    
    # Adjust for the start point
    x = x + start_point[0]
    
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
    """
    Transform the trajectory by translating and rotating.

    Parameters:
    - trajectory: numpy array of shape (n, 5) where n is the number of points.
    - translation: tuple (tx, ty) specifying the translation vector.
    - theta: rotation angle in radians.

    Returns:
    - Transformed trajectory.
    """
    # Translation
    translated_positions = trajectory[:, :2]

    # Rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    # Rotate positions
    rotated_positions = translated_positions @ rotation_matrix.T + np.array(translation)

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
    tfinal = 10  # Adjust this based on the desired length of the trajectory
    dt = 0.01
    translation = (0, 0)
    theta = np.pi / 4  # Rotation angle in radians (e.g., pi/4 for 45 degrees)
    num_s_shapes = 1  # Number of S shapes (sine wave cycles)
    start_point = (0, 0)
    amplitude = 5  # Amplitude of the sine wave
    wavelength = 20  # Wavelength of the sine wave
    velocity = 1.0  # Desired constant velocity (distance per time step)
    
    positions = generate_snake_s_shape_trajectory(tfinal, dt, translation, theta, num_s_shapes, start_point, amplitude, wavelength, velocity)

    # Plot the generated snake-like S-shape trajectory
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))

    axs.plot(positions[:, 0], positions[:, 1], 'b-')
    axs.set_title(f"Rotated Snake-Like S-Shape Trajectory ({num_s_shapes} cycles)")
    axs.set_xlabel("X Position")
    axs.set_ylabel("Y Position")
    axs.grid(True)
    axs.axis('equal')

    plt.tight_layout()
    plt.show()
