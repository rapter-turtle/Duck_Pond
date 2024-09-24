% Double Integrator System with HOSMO and Control Input (Extended to x, y dynamics)

% Parameters
dt = 0.05;              % Time step
t_final = 300;         % Final time
t = 0:dt:t_final;      % Time vector

n = length(t);         % Number of time steps

% State vectors: [x; y; x_dot; y_dot]
x_true = zeros(4, n);        % True states [x; y; x_dot; y_dot]
x_hat = zeros(4, n);         % Estimated states [x_hat; y_hat; x_dot_hat; y_dot_hat]
z = zeros(3, 2, n);          % Correction terms z1, z2, z3 for both x and y
e = zeros(2, n);             % Errors [e_x; e_y]
u = zeros(2, n);             % Control inputs [u_x; u_y]
disturbance_x = 4 * sin(0.4*t); % Disturbance for x dynamics
disturbance_y = 3 * cos(0.5*t); % Disturbance for y dynamics

integ = zeros(2, n);         % Integral of the sign(e)
integ_s = zeros(2, n);       % Integral of the sliding variable sign

% Gains
k1 = 4;
k2 = 3;
k3 = 0.01;
c1 = 1;
lambda1 = 2;
lambda2 = 2;

% Initial Conditions
x_true(:,1) = [5; 0; 0; 0];     % Initial condition for true states [x; y; x_dot; y_dot]
x_hat(:,1) = [0; 0; 0; 0];      % Initial condition for estimated states

nom_x = [5; 0; 0; 0];            % Nominal true states [x; y; x_dot; y_dot]

% LQR Design for Nominal Control Input
A = [0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0];
B = [0 0; 0 0; 1 0; 0 1];
Q = diag([10, 10, 1, 1]);  % State weighting matrix
R = diag([1, 1]);          % Control weighting matrix
K = lqr(A, B, Q, R);       % LQR gain

radius = 5;  % Radius of the circle
omega = 0.02; % Angular velocity for circular motion

% Simulation loop
for k = 1:n-1
    % Define time-varying desired state for circular motion
    desired_state = [radius * cos(omega * t(k)); radius * sin(omega * t(k)); 
                     -radius * omega * sin(omega * t(k)); radius * omega * cos(omega * t(k))];
    
    % Error calculation
    e(:,k) = x_true(1:2, k) - nom_x(1:2, k) - x_hat(1:2, k);
    
    % Correction terms for x and y
    for i = 1:2
        z(1,i,k) = k1 * abs(e(i,k))^(2/3) * sign(e(i,k));
        z(2,i,k) = k2 * abs(e(i,k))^(1/3) * sign(e(i,k));
        z(3,i,k) = k3 * sign(e(i,k));
        
        % Integral term update
        integ(i,k+1) = integ(i,k) + dt * sign(e(i,k));
        integ_s(i,k+1) = integ_s(i,k) + dt * sign(x_hat(2+i, k)); % using x_dot_hat or y_dot_hat as sliding variables
    end
    
    % Update observer dynamics for x and y
    x_hat(1:2, k+1) = x_hat(1:2, k) + (x_hat(3:4, k) + z(1,:,k)')*dt;
    x_hat(3:4, k+1) = x_hat(3:4, k) + (u(:,k) + z(2,:,k)')*dt;
    
    % Control input based on sliding mode control
    u(:,k) = -c1*x_hat(3:4, k) - z(2,:,k)' - integ(:,k).*k3 - lambda1*sqrt(abs(x_hat(3:4, k))) .* sign(x_hat(3:4, k)) - lambda2 * integ_s(:,k);
    % u(:,k) = [0;0];

    % Nominal Control Input using LQR
    u_nom(:,k) = -K * (nom_x(:,k) - desired_state);

    % System dynamics
    x_true(1, k+1) = x_true(1, k) + dt * x_true(3, k);
    x_true(2, k+1) = x_true(2, k) + dt * x_true(4, k);
    x_true(3, k+1) = x_true(3, k) + dt * (u(1,k) + u_nom(1,k) + disturbance_x(k));
    x_true(4, k+1) = x_true(4, k) + dt * (u(2,k) + u_nom(2,k) + disturbance_y(k));

    % Nominal dynamics (assuming no disturbance)
    nom_x(1:2, k+1) = nom_x(1:2, k) + dt * nom_x(3:4, k);
    nom_x(3:4, k+1) = nom_x(3:4, k) + dt * u_nom(:,k);
end

% Plot (x, y) trajectory, disturbance, and control inputs

figure;

% Plot (x, y) trajectory
subplot(3, 1, 1);
plot(x_true(1,:), x_true(2,:), 'LineWidth', 2);
hold on;
plot(nom_x(1,:), nom_x(2,:), '--', 'LineWidth', 2);
title('(x, y) Trajectory');
xlabel('x');
ylabel('y');
legend('True Trajectory', 'Nominal Trajectory');
grid on;

% Plot disturbances over time
subplot(3, 1, 2);
plot(t, disturbance_x, 'r', 'LineWidth', 2);
hold on;
plot(t, disturbance_y, 'b--', 'LineWidth', 2);
title('Disturbances vs. Time');
xlabel('Time (s)');
ylabel('Disturbance');
legend('Disturbance_x', 'Disturbance_y');
grid on;

% Plot control inputs over time
subplot(3, 1, 3);
plot(t, u(1,:), 'r', 'LineWidth', 2);
hold on;
plot(t, u(2,:), 'b--', 'LineWidth', 2);
title('Control Inputs u\_x and u\_y vs. Time');
xlabel('Time (s)');
ylabel('Control Input');
legend('u\_x', 'u\_y');
grid on;

sgtitle('Trajectory, Disturbances, and Control Inputs');
