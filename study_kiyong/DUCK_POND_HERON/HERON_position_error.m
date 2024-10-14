% File names
filenames = {'100800_1_nominal_hp.xlsx', '100800_2_L1_hp.xlsx', '100825_2_DOB_hp.xlsx'};
labels = {'Nominal', 'L1', 'DOB'};
colors = {'b-', 'r-', 'g-'};

% Initialize the figure
figure;
hold on;
xlabel('Time (s)');
ylabel('Position Error (m)');
title('Position Error Over Time');
grid on;

% Time step (adjust if different)
dt = 0.1;

% Loop through each file
for i = 1:length(filenames)
    % Load the data
    data = readtable(filenames{i});
    
    % Extract current and reference x, y positions
    current_x = data.ekf_estimated0;  % Current x position
    current_y = data.ekf_estimated1;  % Current y position
    ref_x = data.refx0;  % Reference x position
    ref_y = data.refy0;  % Reference y position
    
    % Calculate Euclidean distance error
    position_error = sqrt((current_x - ref_x).^2 + (current_y - ref_y).^2);
    
    % Generate time vector
    time = (0:length(position_error)-1) * dt;
    
    % Plot the position error
    plot(time, position_error, colors{i}, 'DisplayName', labels{i}, 'LineWidth', 1.5);
end

% Add legend
legend('Location', 'northeast');
hold off;
