% Given parameters
m = 4.5096 * 1000000;
Xu = -51380;

% Define the matrix A
As = [0, 1; 0, Xu/m];
K = -[5000, 300000];
% Define the matrix B
B1 = [0; 1/m];
B2 = [1; 0];
L = [1, 0; 0, 1];
A = As + B1*K;

% Simulation parameters
dt = 0.01; % Time step (seconds)
total_time = 500; % Total simulation time (seconds)
num_steps = total_time / dt; % Number of simulation steps

% Initialize state vectors
x = zeros(2, num_steps); % State vector [position; velocity]
u = zeros(1, num_steps); % Control input vector
% d = 1.0 * ones(1, num_steps); % Disturbance vector (e.g., Gaussian noise)
time = 0:dt:(total_time-dt); % Time vector
d = sin(0.1*time); % Disturbance vector (sine function)

estim_x = zeros(2, num_steps); % State vector [position; velocity]
um = zeros(1, num_steps);

% Initial condition
x(:, 1) = [0; 0]; % Initial position and velocity
estim_x(:, 1) = x(:, 1);

cutoff = 5;
um_dot = 0;
% Simulation loop (Euler method)
for k = 2:num_steps-1
    x_error = estim_x(:, k) - x(:, k);

    gain = -1;
    pi = (1/gain)*(exp(gain*dt)-1);
    um(k) = -exp(gain*dt)*x_error(1)/pi;
    um_dot = (um(k)-um(k-1))/dt;
    con = m*um_dot - (Xu+K(2))*um(k);
    
    % if con > 1000000
    %     con = 1000000;
    % elseif con < -1000000
    %     con = -1000000;
    % end

    % u(k) = 0;
    u(k) = u(k-1)*exp(-cutoff*dt) - con*(1-exp(-cutoff*dt));

    % Compute the state derivative with disturbance
    dx = A * x(:, k) + B1 * u(k) + B2*d(k);
    estim_dx = A * estim_x(:, k) + B1 *u(k) + B2*um(k) - L*(x_error);
    
    % Update the state using Euler method
    x(:, k+1) = x(:, k) + dx * dt;
    estim_x(:, k+1) = estim_x(:, k) + estim_dx * dt;
end

% Time vector for plotting
time = 0:dt:(total_time-dt);

% Plotting the results
figure;
subplot(4,1,1);
plot(time, x(1, :));
title('Position real vs Time');
xlabel('Time (s)');
ylabel('Position real');

subplot(4,1,2);
plot(time, estim_x(1, :));
title('Position estim vs Time');
xlabel('Time (s)');
ylabel('Position estim');

subplot(4,1,3);
plot(time, um);
title('Disturbance vs Time');
xlabel('Time (s)');
ylabel('Disturbance');

subplot(4,1,4);
plot(time, u);
title('Control vs Time');
xlabel('Time (s)');
ylabel('Control');
