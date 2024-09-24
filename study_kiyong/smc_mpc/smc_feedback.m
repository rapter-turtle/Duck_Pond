% Double Integrator System with HOSMO and Control Input

% Parameters
dt = 0.1;              % Time step
t_final = 10;           % Final time
t = 0:dt:t_final;       % Time vector

n = length(t);          % Number of time steps
x1 = zeros(1, n);       % True state x1
x2 = zeros(1, n);       % True state x2
x1_hat = zeros(1, n);   % Estimated state x1_hat
x2_hat = zeros(1, n);
x3_hat = zeros(1, n);% Estimated state x2_hat
s_hat = zeros(1, n);    % Estimated sliding variable s_hat
e1 = zeros(1, n);       % Error e1 = x1 - x1_hat
e2 = zeros(1, n);       % Error e2 = x2 - x2_hat
z1 = zeros(1, n);       % Correction term z1
z2 = zeros(1, n);       % Correction term z2
z3 = zeros(1, n);       % Correction term z3
u = zeros(1, n);        % Control input
disturbance = 4 * sin(0.3*t); % Disturbance (example sinusoidal disturbance)
integ = zeros(1, n);    % Integral of the sign(e1)
integ_s = zeros(1, n);

% Gains
k1 = 4;
k2 = 3;
k3 = 0.01;
c1 = 1;
lambda1 = 2;
lambda2 = 2;

% Initial Conditions
x1(1) = 0;              % Initial condition for x1
x2(1) = 0;              % Initial condition for x2
x1_hat(1) = 0;          % Initial condition for x1_hat
x2_hat(1) = 0;          % Initial condition for x2_hat
x3_hat(1) = 0;

x1d = 1;
x2d = 0;

nom_x1 = zeros(1, n);       % True state x1
nom_x2 = zeros(1, n);
u_nom = 0.5;


% Simulation loop
for k = 1:n-1
    % Error calculation
    e1(k) = x1(k) - nom_x1(k) - x1_hat(k);
    e2(k) = x2(k) - nom_x2(k) - x2_hat(k);
    
    % Correction terms
    z1(k) = k1 * abs(e1(k))^(2/3) * sign(e1(k));
    z2(k) = k2 * abs(e1(k))^(1/3) * sign(e1(k));
    z3(k) = k3 * sign(e1(k));
    
    % Integral term update
    integ(k+1) = integ(k) + dt * sign(e1(k));
    integ_s(k+1) = integ_s(k) + dt * sign(s_hat(k));
    
    % Update observer dynamics
    x1_hat(k+1) = x1_hat(k) + (x2_hat(k) + z1(k))*dt;
    x2_hat(k+1) = x2_hat(k) + (x3_hat(k) + u(k) + z2(k))*dt;
    x3_hat(k+1) = x3_hat(k) + z3(k)*dt;

    s_hat(k+1) = s_hat(k) + dt*(c1*x2_hat(k) + c1*e2(k) + u(k) + z2(k) + integ(k)*k3);


    % Control input based on sliding mode control
    u(k) = -c1*x2_hat(k) - z2(k) - integ(k)*k3 - lambda1*sqrt(abs(s_hat(k))) * sign(s_hat(k)) - lambda2 * integ_s(k);
    % u(k) = 0;
    % System dynamics
    x1(k+1) = x1(k) + dt * x2(k);
    x2(k+1) = x2(k) + dt * (u(k) + u_nom + disturbance(k));
    % x2(k+1) = x2(k) + dt * (disturbance(k));

    nom_x1(k+1) = nom_x1(k) + dt * nom_x2(k);
    nom_x2(k+1) = nom_x2(k) + dt * u_nom;

end
% Plot results
figure;
subplot(6, 1, 1);
plot(t, x1, 'LineWidth', 2);
hold on;
plot(t, x1_hat, '--', 'LineWidth', 2);
title('x1 and Estimated x1\_hat vs. Time');
xlabel('Time (s)');
ylabel('x1, x1\_hat');
legend('x1', 'x1\_hat');

subplot(6, 1, 2);
plot(t, x2, 'LineWidth', 2);
hold on;
plot(t, x2_hat, '--', 'LineWidth', 2);
title('x2 and Estimated x2\_hat vs. Time');
xlabel('Time (s)');
ylabel('x2, x2\_hat');
legend('x2', 'x2\_hat');

subplot(6, 1, 3);
plot(t, e1, 'LineWidth', 2);
title('Error e1 vs. Time');
xlabel('Time (s)');
ylabel('e1');

subplot(6, 1, 4);
plot(t, e2, 'LineWidth', 2);
title('Error e2 vs. Time');
xlabel('Time (s)');
ylabel('e2');

subplot(6, 1, 5);
plot(t, u, 'r', 'LineWidth', 2);
hold on;
plot(t, disturbance, '--', 'LineWidth', 2);
title('Control Input u vs. Time');
xlabel('Time (s)');
ylabel('Control Input u');

subplot(6, 1, 6);
plot(t, s_hat, 'r', 'LineWidth', 2);
title('s_hat vs. Time');
xlabel('Time (s)');
ylabel('s_hat');

sgtitle('Double Integrator System with HOSMO, Control Input, and Errors');
