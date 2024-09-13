% System matrices
A = [0, 0, 1, 0;
     0, 0, 0, 1;
     0, 0, 0, 0;
     0, 0, 0, 0];

B = [0, 0;
     0, 0;
     1, 0;
     0, 1];

% Define the cost matrices for state and input
Q = [1 0 0 0; 
     0 1 0 0; 
     0 0 1 0; 
     0 0 0 1];   % State weighting matrix

R = [1 0; 0 1]; % Control weighting matrix

% Calculate the optimal feedback gain matrix K using LQR
[K, S, E] = lqr(A, B, Q, R);

% Simulation parameters
dt = 0.1;              % Time step
T = 100;                % Total simulation time
n_steps = T / dt;      % Number of time steps
t = 0:dt:T;            % Time vector

% Initial state
x_history = zeros(4, length(t)); % Store state history for plotting
desired_traj_history = zeros(2, length(t));

% Go straight
% x = [3; 2; 0; 0];  % Initial state (position and velocity)
% vx = 0.5;
% vy = 0.5;
% deried_traj_straight = [0;0;vx;vy];

% Go circle
theta = 0;
R = 2;
omega = 0.2;
x = [R; 0; 0; 0];
deried_traj_circle = [R; 0; -R*omega*sin(theta); R*omega*cos(theta)];

% Perform the simulation using a for loop
for i = 1:n_steps
    
    % Go straight
    % deried_traj_straight = deried_traj_straight + [vx*dt;vy*dt;0;0];
    % u = -K * (x - deried_traj_straight);

    % Go circle
    theta = omega*i*dt;
    deried_traj_circle = [R*cos(theta); R*sin(theta); -R*omega*sin(theta); R*omega*cos(theta)];
    u = -K * (x - deried_traj_circle);

    % State update using Euler integration (x_dot = Ax + Bu)
    x_dot = A * x + B * u;
    x = x + x_dot * dt;  % Euler integration
    
    % Store the state for plotting
    x_history(:, i) = x;

    desired_traj_history(:, i) = deried_traj_circle(1:2);
end

% % Plot results
% figure;
% subplot(2, 1, 1);
% plot(t(1:n_steps), x_history(1, 1:n_steps), 'r', 'LineWidth', 1.5); hold on;
% plot(t(1:n_steps), x_history(2, 1:n_steps), 'b', 'LineWidth', 1.5);
% xlabel('Time (s)');
% ylabel('Position');
% legend('x_1 (Position 1)', 'x_2 (Position 2)');
% title('State Evolution: Position');
% grid on;
% 
% subplot(2, 1, 2);
% plot(t(1:n_steps), x_history(3, 1:n_steps), 'r', 'LineWidth', 1.5); hold on;
% plot(t(1:n_steps), x_history(4, 1:n_steps), 'b', 'LineWidth', 1.5);
% xlabel('Time (s)');
% ylabel('Velocity');
% legend('x_3 (Velocity 1)', 'x_4 (Velocity 2)');
% title('State Evolution: Velocity');
% grid on;

% Plot the XY trajectory
figure;
plot(x_history(1, 1:n_steps), x_history(2, 1:n_steps), 'r', 'LineWidth', 1.5); hold on;
plot(desired_traj_history(1, 1:n_steps), desired_traj_history(2, 1:n_steps), 'b--', 'LineWidth', 1.5);
xlabel('x-position');
ylabel('y-position');
title('XY Trajectory of the System');
legend('Actual Trajectory', 'Desired Circular Trajectory');
grid on;