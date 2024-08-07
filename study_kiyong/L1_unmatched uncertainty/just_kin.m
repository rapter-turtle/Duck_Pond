clear 


%%%%%%%%%%%%%%%%%%%%%%%%%%% State estimate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A_estim = [0, 0, 1, 0;
     0, 0, 0, 1;
     0, 0, 0, 0;
     0, 0, 0, 0];

B_estim = [0, 0;
     0, 0;
     1, 0;
     0, 1];

Bum_estim = [1, 0;
     0, 1;
     0, 0;
     0, 0];

L = diag([1, 1, 1, 1]);

% Define the state weighting matrix Q and control weighting matrix R
Q = diag([1, 1, 1, 1]); % Adjust the values based on your requirements
R = diag([1, 1]);          % Adjust the values based on your requirements

% Calculate the LQR gain K
[K, S, e] = lqr(A_estim, B_estim, Q, R);

cutoff = 20;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation parameters
dt = 0.01; % Time step (seconds)
total_time = 100; % Total simulation time (seconds)
num_steps = total_time / dt; % Number of simulation steps

% Initialize state vectors
x = zeros(4, num_steps); % State vector [position; velocity]
u = zeros(2, num_steps); % Control input vector
estim_x = zeros(4, num_steps);
virtual_state = zeros(4, num_steps);
con = zeros(4, num_steps);
virtual_u = zeros(2, num_steps);
um = zeros(4, num_steps);
um_dot = 0;
filtered_um = zeros(4, num_steps);

time = 0:dt:(total_time-dt); % Time vector

% Initial condition
x(:, 1) = [0; 0; 0; 0]; % Initial position and velocity
estim_x(:, 1) = [0; 0; 0; 0];


% Simulation loop (Euler method)
for k = 2:num_steps-1
    
    % L1 control input
    x_error = estim_x(:, k) - x(:,k);
    gain = -1;
    pi = (1/gain)*(exp(gain*dt)-1);
    um(:,k) = -exp(gain*dt)*x_error/pi;
    um_dot = (um(:,k)-um(:,k-1))/dt;
    con(1:2,k) = um_dot(1:2) + 1.73*um(1:2,k);
    con(3:4,k) = um(3:4,k);
    filtered_um(:,k) = filtered_um(:,k-1)*exp(-cutoff*dt) - con(:,k)*(1-exp(-cutoff*dt));
    

    u = [sin(time(k)); cos(time(k))];
    Am = A_estim - B_estim*K;
    estim_dx = Am*estim_x(:,k) + B_estim*(filtered_um(1:2,k) + filtered_um(3:4,k)) + B_estim*u + +B_estim*um(3:4,k) + Bum_estim*um(1:2,k) - L*(x_error);

    um_disturbance = [1; 2.0];
    m_disturbance = [10; 20.0];
    dx = Am*x(:,k) + B_estim*(filtered_um(1:2,k) + filtered_um(3:4,k)) + B_estim*u + Bum_estim*um_disturbance + B_estim*m_disturbance;

    x(:, k+1) = x(:, k) + dx * dt;
    estim_x(:, k+1) = estim_x(:, k) + estim_dx * dt;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
subplot(4,1,1);
plot(time, x(1,:));
title('Headpoint x');
xlabel('Time (s)');
ylabel('Headpoint x');

subplot(4,1,2);
plot(time, x(2,:));
title('Headpoint y');
xlabel('Time (s)');
ylabel('Headpoint y');

subplot(4,1,3);
plot(time,um);
title('Disturbance');
xlabel('Time (s)');
ylabel('Disturbance');


subplot(4,1,4);
plot(time, u(1,:));
% plot(time, u(2,:));
title('Control input');
xlabel('Time (s)');
ylabel('Control');

