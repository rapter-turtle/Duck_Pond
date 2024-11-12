clear

% Given matrices and constants
A = [-1, 2; -3, 4];
B = [0.5;-2];
Bw = [0;1];
rho = 2.0; % Given value of rho

% Define YALMIP variables
X = sdpvar(2,2);         % Symmetric matrix variable X
Y = sdpvar(1,2);         % Rectangular matrix variable Y
% lambda0 = sdpvar(1);     % Scalar variable lambda0
muu = sdpvar(1);         % Scalar variable muu
alpha_min = sdpvar(1);   % Variable for the smallest eigenvalue
% lambda = sdpvar(1);      % Variable for lambda
P = sdpvar(2,2);         % Auxiliary variable for P = X^(-1)

% Initialize constraints
Constraints = [X >= 1e-12 * eye(2)]; % Relax positive definiteness if needed
Constraints = [Constraints,  muu >= 0, alpha_min >= 0];

% Schur complement constraint to enforce P = X^(-1)
Constraints = [Constraints, [X, eye(2); eye(2), P] >= 0];

% Set alpha_min as the smallest eigenvalue of P
Constraints = [Constraints, P >= alpha_min * eye(2)];

% Main LMI condition
LMI = [ (A*X + B*Y)' + A*X + B*Y + 5*X, Bw; 
        Bw', -muu];
Constraints = [Constraints, LMI <= 0]; % Add LMI constraint

% Constraints = [Constraints, lambda - lambda0 <= 0]
% Add the new constraint based on the inequality
% Constraints = [Constraints, rho <= (lambda0 - lambda) * alpha_min / 2];

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
    % lambda0_opt = value(lambda0);
    muu_opt = value(muu);
    alpha_min_opt = value(alpha_min);
    % lambda_opt = value(lambda);

    disp('Optimal values found:')
    disp('X = ')
    disp(X_opt)
    disp('Y = ')
    disp(Y_opt)
    % disp(['lambda0 = ' num2str(lambda0_opt)])
    disp(['muu = ' num2str(muu_opt)])
    disp(['alpha_min (smallest eigenvalue of P) = ' num2str(alpha_min_opt)])
    % disp(['lambda = ' num2str(lambda_opt)])
else
    disp('The problem is infeasible. Adjust constraints or review model formulation.')
end

% K = Y_opt*inv(X_opt)

% Given system matrices and feedback control
A = [-1, 2; -3, 4];
B = [0.5; -2];
Bw = [0; 1];
X_opt = value(X);
Y_opt = value(Y);
K = Y_opt / X_opt;  % Compute the feedback gain

% Define the closed-loop system matrix
A_cl = A + B * K;

% Define time span for simulation
tspan = [0 10];  % 10 seconds simulation time

% Define disturbance function (varying between -1 and 1)
disturbance_min = -1;
disturbance_max = 1;

% Initial condition for the state
x0 = [9; 5];  % Initial state (can adjust as desired)

% Simulation settings
dt = 0.01;  % Time step
time = tspan(1):dt:tspan(2);

% Pre-allocate state and disturbance arrays
x = zeros(2, length(time));
x(:,1) = x0;
disturbance = zeros(1, length(time));

% Random disturbance generation within the range [-1, 1]
rng(0);  % For reproducibility
disturbance = disturbance_min + (disturbance_max - disturbance_min) * (2 * rand(1, length(time)) - 1);

% Simulate the system
for i = 1:length(time)-1
    % Compute x_dot
    x_dot = A_cl * x(:,i) + Bw * disturbance(i);
    
    % Update the state using Euler integration
    x(:,i+1) = x(:,i) + x_dot * dt;
end

% Plot results
figure;
subplot(3,1,1);
plot(time, x(1,:), 'b', 'LineWidth', 1.5);
title('State x_1 vs. Time');
xlabel('Time (s)');
ylabel('x_1');

subplot(3,1,2);
plot(time, x(2,:), 'r', 'LineWidth', 1.5);
title('State x_2 vs. Time');
xlabel('Time (s)');
ylabel('x_2');

subplot(3,1,3);
plot(time, disturbance, 'k', 'LineWidth', 1.5);
title('Disturbance vs. Time');
xlabel('Time (s)');
ylabel('Disturbance');
ylim([disturbance_min-0.5, disturbance_max+0.5]);

sgtitle('System Response with Disturbance Input');
