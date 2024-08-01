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

cutoff = 30;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation parameters
dt = 0.01; % Time step (seconds)
total_time = 1000; % Total simulation time (seconds)
num_steps = total_time / dt; % Number of simulation steps

% Initialize state vectors
x = zeros(4, num_steps); % State vector [position; velocity]
u = zeros(2, num_steps); % Control input vector
estim_x = zeros(4, num_steps);
virtual_state = zeros(4, num_steps);
virtual_u = zeros(2, num_steps);
um = zeros(1, num_steps);
um_dot = 0;
filtered_um = zeros(1, num_steps);

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
    um(k) = -exp(gain*dt)*x_error(1)/pi;
    um_dot = (um(k)-um(k-1))/dt;
    con = um_dot + 1.73*um(k);
    filtered_um(k) = filtered_um(k-1)*exp(-cutoff*dt) - con*(1-exp(-cutoff*dt));

    Am = A_estim - B_estim*K;
    estim_dx = Am*estim_x(:,k) + B_estim*[filtered_um(k);0] + Bum_estim*[um(k) ; 0] - L*(x_error);

    disturbance = [10; 0.0];
    dx = Am*x(:,k) + B_estim*[filtered_um(k);0] + Bum_estim*disturbance;

    x(:, k+1) = x(:, k) + dx * dt;
    estim_x(:, k+1) = estim_x(:, k) + estim_dx * dt;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
subplot(4,1,1);
plot(time, x(1, :));
title('Position real vs Time');
xlabel('Time (s)');
ylabel('Position real');

subplot(4,1,2);
plot(time, um);
title('Disturbance vs Time');
xlabel('Time (s)');
ylabel('Disturbance');

subplot(4,1,3);
plot(time, u);
title('Control vs Time');
xlabel('Time (s)');
ylabel('Control');


subplot(4,1,4);
plot(time, filtered_um);
title('filtered u');
xlabel('Time (s)');
ylabel('Control');
