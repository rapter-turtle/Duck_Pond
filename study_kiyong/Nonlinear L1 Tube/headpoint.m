clear

% Define continuous-time matrices
A_continuous = [0, 0, 1, 0; 
     0, 0, 0, 1; 
     0, 0, 0, 0; 
     0, 0, 0, 0];
B_continuous = [0, 0; 
     0, 0; 
     1, 0; 
     0, 1];

% Sampling time (choose an appropriate value, e.g., Ts = 0.1 seconds)
Ts = 0.1;

% Create a continuous-time state-space model
sys_continuous = ss(A_continuous, B_continuous, [], []);

% Convert the continuous-time model to a discrete-time model
sys_discrete = c2d(sys_continuous, Ts);

% Extract the discrete-time matrices
A = sys_discrete.A;
B = sys_discrete.B;
Bw = B;

% Define YALMIP variables
X = sdpvar(4,4);         % Symmetric matrix variable X
Y = sdpvar(2,4);         % Rectangular matrix variable Y
muu = sdpvar(1);         % Scalar variable muu
alpha_min = sdpvar(1);   % Variable for the smallest eigenvalue
P = sdpvar(4,4);         % Auxiliary variable for P = X^(-1)

% Initialize constraints
Constraints = [X >= 1e-12 * eye(4)]; % Relax positive definiteness if needed
Constraints = [Constraints,  muu >= 0, alpha_min >= 0];

% Schur complement constraint to enforce P = X^(-1)
Constraints = [Constraints, [X, eye(4); eye(4), P] >= 0];

% Set alpha_min as the smallest eigenvalue of P
Constraints = [Constraints, P >= alpha_min * eye(4)];

% Main LMI condition
LMI = [ (A*X + B*Y)' + A*X + B*Y + 1*X, Bw; 
        Bw', -muu * eye(2)];
Constraints = [Constraints, LMI <= 0]; % Add LMI constraint

% Objective: Maximize lambda0 (changed to maximize lambda0, as we previously minimized -lambda0)
Objective = [];

% Solver settings
options = sdpsettings('solver', 'sdpt3', 'verbose', 1); % Verbose output for debugging

% Solve the problem
sol = optimize(Constraints, Objective, options);

% Check feasibility
if sol.problem == 0
    % Retrieve and display results
    X_opt = value(X);
    Y_opt = value(Y);
    muu_opt = value(muu);
    alpha_min_opt = value(alpha_min);

    disp('Optimal values found:')
    disp('X = ')
    disp(X_opt)
    disp('Y = ')
    disp(Y_opt)
    disp(['muu = ' num2str(muu_opt)])
    disp(['alpha_min (smallest eigenvalue of P) = ' num2str(alpha_min_opt)])
else
    disp('The problem is infeasible. Adjust constraints or review model formulation.')
end

% K = Y_opt / X_opt

% Given system matrices and feedback control
% Define the closed-loop system matrix
A_cl = A + B * (Y_opt / X_opt);

% Define time span for simulation
tspan = [0 10];  % 10 seconds simulation time

% Define disturbance function (varying between -1 and 1)
disturbance_min = -1;
disturbance_max = 1;

% Initial condition for the state
x0 = [9; 5; 3; 5];  % Initial state (can adjust as desired)

% Simulation settings
dt = 0.01;  % Time step
time = tspan(1):dt:tspan(2);

% Pre-allocate state and disturbance arrays
x = zeros(4, length(time));
x(:,1) = x0;
disturbance = zeros(2, length(time));

% Random disturbance generation within the range [-1, 1]
rng(0);  % For reproducibility
disturbance = disturbance_min + (disturbance_max - disturbance_min) * (2 * rand(2, length(time)) - 1);

% Simulate the system
for i = 1:length(time)-1
    % Compute x_dot
    x_dot = A_cl * x(:,i) + Bw * disturbance(:,i);

    % Update the state using Euler integration
    x(:,i+1) = x(:,i) + x_dot * dt;
end

% Plot results
figure;
subplot(5,1,1);
plot(time, x(1,:), 'b', 'LineWidth', 1.5);
title('State x_1 vs. Time');
xlabel('Time (s)');
ylabel('x_1');

subplot(5,1,2);
plot(time, x(2,:), 'r', 'LineWidth', 1.5);
title('State x_2 vs. Time');
xlabel('Time (s)');
ylabel('x_2');

subplot(5,1,3);
plot(time, x(3,:), 'g', 'LineWidth', 1.5);
title('State x_3 vs. Time');
xlabel('Time (s)');
ylabel('x_3');

subplot(5,1,4);
plot(time, x(4,:), 'm', 'LineWidth', 1.5);
title('State x_4 vs. Time');
xlabel('Time (s)');
ylabel('x_4');

subplot(5,1,5);
plot(time, disturbance, 'k', 'LineWidth', 1.5);
title('Disturbance vs. Time');
xlabel('Time (s)');
ylabel('Disturbance');
ylim([disturbance_min-0.5, disturbance_max+0.5]);

sgtitle('System Response with Disturbance Input');
