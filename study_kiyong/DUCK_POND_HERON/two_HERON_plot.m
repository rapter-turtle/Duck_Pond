% Load the Excel files
filename1 = '100800_2_L1_hp.xlsx';
data1 = readtable(filename1);

filename2 = '100825_2_DOB_hp.xlsx';
data2 = readtable(filename2);

% Plot offset
offset_x = 353152;
offset_y = 4026032;

% Extract x, y, and heading data for first USV from the 'ekf_estimated' columns
x1 = data1.ekf_estimated0 - offset_x;
y1 = data1.ekf_estimated1 - offset_y;
heading1 = data1.ekf_estimated2;
x1_head = data1.ekf_estimated0 - offset_x + cos(heading1);
y1_head = data1.ekf_estimated1 - offset_y + sin(heading1);

% Extract x, y, and heading data for second USV from the 'ekf_estimated' columns
x2 = data2.ekf_estimated0 - offset_x;
y2 = data2.ekf_estimated1 - offset_y;
heading2 = data2.ekf_estimated2;

% Extract trajectory generation parameters
tfinal = data2.mpc_traj0(1);
dt = data2.mpc_traj1(1);
theta = data2.mpc_traj2(1);
num_s_shapes = 3;
start_x = data2.mpc_traj4(1);
start_y = data2.mpc_traj5(1);
amplitude = data2.mpc_traj6(1);
wavelength = data2.mpc_traj7(1);
velocity = data2.mpc_traj8(1);

% Generate the reference trajectory
ref = generate_snake_s_shape_trajectory(tfinal, dt, [0, 0], theta, num_s_shapes, [start_x, start_y], amplitude, wavelength, velocity);
trajectory_angle = theta;
rotation_matrix = [cos(-trajectory_angle), -sin(-trajectory_angle); sin(-trajectory_angle), cos(-trajectory_angle)];

% Adjust trajectory and USV coordinates
ref_adjusted = ref(:,1:2) - ref(1,1:2);
ref_adjusted = (rotation_matrix * ref_adjusted')';
usv_coords1_adjusted = (rotation_matrix * ([x1, y1] - ref(1,1:2))')';
usv_head_coords1_adjusted = (rotation_matrix * ([x1_head, y1_head] - ref(1,1:2))')';
usv_coords2_adjusted = (rotation_matrix * ([x2, y2] - ref(1,1:2))')';

% Set up the figure for animation
% figure;
figure('Position', [100, 100, 1200, 600]); % Adjust the width and height as desired
hold on;
plot(ref_adjusted(:,1), ref_adjusted(:,2), 'b--', 'LineWidth', 1.5); % Reference trajectory
axis equal;
grid on;
xlabel('X Position');
ylabel('Y Position');
title('USV Position, Heading, and Reference Trajectory');
xlim([-10, 70]);
ylim([-10, 10]);

% USV Shape Parameters
hullWidth = 0.35;
hullLength = 2.0;
boxWidth = 1.3;
boxLength = 1.5;
hullOffset = 0.8;

% Plot initialization for USVs
h_leftHull1 = fill(nan, nan, 'k', 'FaceAlpha', 0.6);
h_rightHull1 = fill(nan, nan, 'k', 'FaceAlpha', 0.6);
h_centerBox1 = fill(nan, nan, 'y', 'FaceAlpha', 0.6);

h_leftHull2 = fill(nan, nan, 'k', 'FaceAlpha', 0.6);
h_rightHull2 = fill(nan, nan, 'k', 'FaceAlpha', 0.6);
h_centerBox2 = fill(nan, nan, 'c', 'FaceAlpha', 0.6);

trajectory_plot1 = plot(nan, nan, 'r', 'LineWidth', 1.2);
trajectory_plot2 = plot(nan, nan, 'm', 'LineWidth', 1.2);

% Dummy elements for legend
legend_yellow = fill(nan, nan, 'y', 'FaceAlpha', 0.6, 'EdgeColor', 'none');
legend_cyan = fill(nan, nan, 'c', 'FaceAlpha', 0.6, 'EdgeColor', 'none');

% Initialize paths for USVs
usv_path_x1 = [];
usv_path_y1 = [];
usv_path_x2 = [];
usv_path_y2 = [];


% Legend for USVs only
legend([legend_yellow, legend_cyan], {'L1 + NMPC', 'NMPC'}, 'Location', 'northeast');
% Animation loop
for k = 1:min(length(usv_coords1_adjusted), length(usv_coords2_adjusted))
    R1 = [cos(heading1(k) - trajectory_angle), -sin(heading1(k) - trajectory_angle);
          sin(heading1(k) - trajectory_angle),  cos(heading1(k) - trajectory_angle)];
    R2 = [cos(heading2(k) - trajectory_angle), -sin(heading2(k) - trajectory_angle);
          sin(heading2(k) - trajectory_angle),  cos(heading2(k) - trajectory_angle)];

    leftHullCoords = [
        -hullLength/2, hullLength/2, hullLength/2 + 0.5, hullLength/2, -hullLength/2;
        -hullWidth, -hullWidth, 0, hullWidth, hullWidth
    ] - [0; hullOffset];

    rightHullCoords = [
        -hullLength/2, hullLength/2, hullLength/2 + 0.5, hullLength/2, -hullLength/2;
        -hullWidth, -hullWidth, 0, hullWidth, hullWidth
    ] + [0; hullOffset];

    centerBoxCoords = [-boxLength/2, boxLength/2, boxLength/2, -boxLength/2;
                       -boxWidth/2, -boxWidth/2, boxWidth/2, boxWidth/2];

    leftHullWorld1 = R1 * leftHullCoords + usv_coords1_adjusted(k,:)';
    rightHullWorld1 = R1 * rightHullCoords + usv_coords1_adjusted(k,:)';
    centerBoxWorld1 = R1 * centerBoxCoords + usv_coords1_adjusted(k,:)';

    leftHullWorld2 = R2 * leftHullCoords + usv_coords2_adjusted(k,:)';
    rightHullWorld2 = R2 * rightHullCoords + usv_coords2_adjusted(k,:)';
    centerBoxWorld2 = R2 * centerBoxCoords + usv_coords2_adjusted(k,:)';

    set(h_leftHull1, 'XData', leftHullWorld1(1, :), 'YData', leftHullWorld1(2, :));
    set(h_rightHull1, 'XData', rightHullWorld1(1, :), 'YData', rightHullWorld1(2, :));
    set(h_centerBox1, 'XData', centerBoxWorld1(1, :), 'YData', centerBoxWorld1(2, :));

    set(h_leftHull2, 'XData', leftHullWorld2(1, :), 'YData', leftHullWorld2(2, :));
    set(h_rightHull2, 'XData', rightHullWorld2(1, :), 'YData', rightHullWorld2(2, :));
    set(h_centerBox2, 'XData', centerBoxWorld2(1, :), 'YData', centerBoxWorld2(2, :));

    % usv_path_x1 = [usv_path_x1, usv_coords1_adjusted(k,1)];
    % usv_path_y1 = [usv_path_y1, usv_coords1_adjusted(k,2)];
    % set(trajectory_plot1, 'XData', usv_path_x1, 'YData', usv_path_y1);

    usv_path_x1 = [usv_path_x1, usv_head_coords1_adjusted(k,1)];
    usv_path_y1 = [usv_path_y1, usv_head_coords1_adjusted(k,2)];
    set(trajectory_plot1, 'XData', usv_path_x1, 'YData', usv_path_y1);    

    usv_path_x2 = [usv_path_x2, usv_coords2_adjusted(k,1)];
    usv_path_y2 = [usv_path_y2, usv_coords2_adjusted(k,2)];
    set(trajectory_plot2, 'XData', usv_path_x2, 'YData', usv_path_y2);

    pause(0.05);
end






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


