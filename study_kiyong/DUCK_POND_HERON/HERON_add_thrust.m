% Load the Excel file
filename = '100800_2_L1_hp.xlsx';
data = readtable(filename);

% Extract thrust data from L1_data columns
n1 = data.L1_data4; % Column 4
n2 = data.L1_data5; % Column 5
thrust1 = data.L1_data6; % Column 6
thrust2 = data.L1_data7; % Column 7

% Define time vector based on the data length and time step
dt = 0.1; % Assuming a time step of 0.1 seconds (adjust if necessary)
time = (0:length(thrust1)-1) * dt;

% Create a new figure for the thrust plots
figure;

% First subplot: Plot for Left NMPC and Left L1+NMPC (n1 and thrust1)
subplot(2, 1, 1); % Creates a 2x1 grid of subplots, this is the first subplot
hold on;
plot(time, thrust1, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Left NMPC');
plot(time, n1, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Left L1+NMPC');
xlabel('Time (s)');
ylabel('Thrust');
title('Left Thrust Inputs Over Time');
legend('Location', 'northeast');
grid on;
hold off;

% Second subplot: Plot for Right NMPC and Right L1+NMPC (n2 and thrust2)
subplot(2, 1, 2); % Second subplot in the 2x1 grid
hold on;
plot(time, thrust2, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Right NMPC');
plot(time, n2, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Right L1+NMPC');
xlabel('Time (s)');
ylabel('Thrust');
title('Right Thrust Inputs Over Time');
legend('Location', 'northeast');
grid on;
hold off;
