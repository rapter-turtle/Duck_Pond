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
T = 20;                % Total simulation time
n_steps = T / dt;      % Number of time steps
t = 0:dt:T;            % Time vector

% Initial state
x = [3; 2; 0; 0];  % Initial state (position and velocity)
x_history = zeros(4, length(t)); % Store state history for plotting


vx = 0.5;
vy = 0.5;
deried_traj = [0;0;vx;vy];

% Perform the simulation using a for loop
for i = 1:n_steps
    % Control law: u = -Kx
    
    deried_traj = deried_traj + [vx*dt;vy*dt;0;0];
    u = -K * (x - deried_traj);
    
    % State update using Euler integration (x_dot = Ax + Bu)
    x_dot = A * x + B * u;
    x = x + x_dot * dt;  % Euler integration
    
    % Store the state for plotting
    x_history(:, i) = x;
end

% Plot results
figure;
subplot(2, 1, 1);
plot(t(1:n_steps), x_history(1, 1:n_steps), 'r', 'LineWidth', 1.5); hold on;
plot(t(1:n_steps), x_history(2, 1:n_steps), 'b', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Position');
legend('x_1 (Position 1)', 'x_2 (Position 2)');
title('State Evolution: Position');
grid on;

subplot(2, 1, 2);
plot(t(1:n_steps), x_history(3, 1:n_steps), 'r', 'LineWidth', 1.5); hold on;
plot(t(1:n_steps), x_history(4, 1:n_steps), 'b', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Velocity');
legend('x_3 (Velocity 1)', 'x_4 (Velocity 2)');
title('State Evolution: Velocity');
grid on;
