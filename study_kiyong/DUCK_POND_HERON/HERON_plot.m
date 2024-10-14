% Load the Excel file
filename = '100800_2.xlsx';
data = readtable(filename);

% Plot offset
offset_x = 353152;
offset_y = 4026032;

% Extract x, y, and heading data from the 'ekf_estimated' columns
x = data.ekf_estimated0 - offset_x;  % Column for x position
y = data.ekf_estimated1 - offset_y;  % Column for y position
heading = data.ekf_estimated2;  % Column for heading (angle in radians or degrees)

% Extract trajectory generation parameters from 'mpc_traj' columns
tfinal = data.mpc_traj0(1);     % Total time for trajectory
dt = data.mpc_traj1(1);         % Time step for trajectory
theta = data.mpc_traj2(1);      % Rotation angle in radians
num_s_shapes = 3;               % Number of S-shapes
start_x = data.mpc_traj4(1);    % Start point x-coordinate
start_y = data.mpc_traj5(1);    % Start point y-coordinate
amplitude = data.mpc_traj6(1);  % Amplitude of the sine wave
wavelength = data.mpc_traj7(1); % Wavelength of the sine wave
velocity = data.mpc_traj8(1);   % Velocity

% Generate the reference trajectory based on extracted parameters
ref = generate_snake_s_shape_trajectory(tfinal, dt, [0, 0], theta, num_s_shapes, [start_x, start_y], amplitude, wavelength, velocity);

% Rotate and shift trajectory to align start at (0,0) and along x-axis
trajectory_angle = theta;%atan2(ref(2,2) - ref(1,2), ref(2,1) - ref(1,1));
rotation_matrix = [cos(-trajectory_angle), -sin(-trajectory_angle); sin(-trajectory_angle), cos(-trajectory_angle)];

% Adjust trajectory
ref_adjusted = ref(:,1:2) - ref(1,1:2); % Shift to (0,0)
ref_adjusted = (rotation_matrix * ref_adjusted')'; % Rotate to align with x-axis

% Adjust USV coordinates
usv_coords = [x, y];
usv_coords_adjusted = usv_coords - ref(1,1:2); % Shift USV to same start as trajectory
usv_coords_adjusted = (rotation_matrix * usv_coords_adjusted')'; % Rotate with trajectory

% Set up the figure for animation
figure;
plot(ref_adjusted(:,1), ref_adjusted(:,2), 'b--', 'LineWidth', 1.5); % Plot the reference trajectory as a background
hold on;
axis equal;
grid on;
xlabel('X Position');
ylabel('Y Position');
title('USV Position, Heading, and Reference Trajectory');
xlim([-10, 70]);
ylim([-10, 10]);

% USV Shape Parameters
hullWidth = 0.35;  % Width of each hull
hullLength = 2.0; % Length of each hull
boxWidth = 1.3;   % Width of the central box
boxLength = 1.5;  % Length of the central box
hullOffset = 0.8; % Offset distance between the center box and each hull

% Plot initialization for the USV
h_leftHull = fill(nan, nan, 'k'); % Left hull (Pentagon)
h_rightHull = fill(nan, nan, 'k'); % Right hull (Pentagon)
h_centerBox = fill(nan, nan, 'y'); % Center box
trajectory_plot = plot(nan, nan, 'r', 'LineWidth', 1.2); % USV trajectory path

% Initialize arrays to store the path of the USV
usv_path_x = [];
usv_path_y = [];

% Animation loop
for k = 1:length(usv_coords_adjusted)
    % Calculate the rotation matrix for heading
    R = [cos(heading(k) - trajectory_angle), -sin(heading(k) - trajectory_angle);
         sin(heading(k) - trajectory_angle),  cos(heading(k) - trajectory_angle)];

    % Define pentagon shape for left and right hulls in local coordinates
    leftHullCoords = [
        -hullLength/2, hullLength/2, hullLength/2 + 0.5, hullLength/2, -hullLength/2;  % X-coordinates
        -hullWidth, -hullWidth, 0, hullWidth, hullWidth  % Y-coordinates
    ] - [0; hullOffset];  % Offset for left hull

    rightHullCoords = [
        -hullLength/2, hullLength/2, hullLength/2 + 0.5, hullLength/2, -hullLength/2;  % X-coordinates
        -hullWidth, -hullWidth, 0, hullWidth, hullWidth  % Y-coordinates
    ] + [0; hullOffset];  % Offset for right hull

    % Define center box in local coordinates
    centerBoxCoords = [-boxLength/2, boxLength/2, boxLength/2, -boxLength/2;
                       -boxWidth/2, -boxWidth/2, boxWidth/2, boxWidth/2];

    % Rotate and translate the shapes based on current x, y, heading
    leftHullWorld = R * leftHullCoords + usv_coords_adjusted(k,:)';
    rightHullWorld = R * rightHullCoords + usv_coords_adjusted(k,:)';
    centerBoxWorld = R * centerBoxCoords + usv_coords_adjusted(k,:)';

    % Update the patches with transformed coordinates
    set(h_leftHull, 'XData', leftHullWorld(1, :), 'YData', leftHullWorld(2, :));
    set(h_rightHull, 'XData', rightHullWorld(1, :), 'YData', rightHullWorld(2, :));
    set(h_centerBox, 'XData', centerBoxWorld(1, :), 'YData', centerBoxWorld(2, :));

    % Store the USV's path
    usv_path_x = [usv_path_x, usv_coords_adjusted(k,1)];
    usv_path_y = [usv_path_y, usv_coords_adjusted(k,2)];
    set(trajectory_plot, 'XData', usv_path_x, 'YData', usv_path_y);

    % Pause to create animation effect
    pause(0.05);  % Adjust pause duration for smoother/faster animation
end

legend('Reference Trajectory', 'USV Heading', 'USV Path');


function ref = generate_snake_s_shape_trajectory(tfinal, dt, translation, theta, num_s_shapes, start_point, amplitude, wavelength, velocity)
    % Generate the time vector
    t = 0:dt:tfinal;

    % Generate a snake-like S-shape using a sine function
    x = linspace(0, num_s_shapes * wavelength, length(t));
    y = amplitude * sin(2 * pi * x / wavelength) + start_point(2);

    % Adjust for the start point
    x = x + start_point(1);

    % Calculate the arc length for each point
    dx = gradient(x);
    dy = gradient(y);
    ds = sqrt(dx.^2 + dy.^2);
    s = cumtrapz(ds);

    % Ensure all values in s, x, and y are finite
    if any(~isfinite(s))
        error('Non-finite values in arc length array s. Check input parameters.');
    end

    % Determine the desired uniform arc length spacing
    total_distance = s(end);
    num_points = floor(total_distance / (velocity * dt));
    s_uniform = linspace(0, total_distance, num_points);

    % Remove any non-finite values before interpolation
    finite_mask = isfinite(s) & isfinite(x) & isfinite(y);
    s = s(finite_mask);
    x = x(finite_mask);
    y = y(finite_mask);

    % Interpolate the x and y positions based on uniform arc length
    x_uniform = interp1(s, x, s_uniform, 'linear', 'extrap');
    y_uniform = interp1(s, y, s_uniform, 'linear', 'extrap');

    % Recalculate derivatives after interpolation
    dx_uniform = gradient(x_uniform, dt);
    dy_uniform = gradient(y_uniform, dt);
    velocity_magnitudes = sqrt(dx_uniform.^2 + dy_uniform.^2);

    % Compute headings and rotational speeds
    headings = atan2(dy_uniform, dx_uniform);
    headings = unwrap(headings);
    rot_speed = gradient(headings, dt);

    % Create the reference trajectory
    ref = [x_uniform', y_uniform', headings', velocity_magnitudes', rot_speed'];

    % Apply translation and rotation transformations
    R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    translated_coords = (R * [x_uniform; y_uniform])';
    ref(:, 1:2) = translated_coords + translation;
end


