clear 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% USV Model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = 3980;
Iz = 19703;
Xu = -50;
Yv = -200;
Yr = 0;
Nv = 0;
Nr = -1281;
m11 = m;
m22 = m; 
m33 = Iz;
l = 3.5;

M = [m11, 0, 0; 0, m22, 0; 0, 0, m33];
D = [Xu, 0, 0; 0, Yv, Yr; 0, Nv, Nr];
M_inv = inv(M);
MD = M_inv*D;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% State estimate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A_estim = [0, 0, 1, 0;
     0, 0, 0, 1;
     0, 0, 0, 0;
     0, 0, 0, 0];

B_estim = [0, 0;
     0, 0;
     1, 0;
     0, 1];

L = diag([1, 1, 1, 1]);

% Define the state weighting matrix Q and control weighting matrix R
Q = diag([1, 1, 1, 1]); % Adjust the values based on your requirements
R = diag([1, 1]);          % Adjust the values based on your requirements

% Calculate the LQR gain K
[K, S, e] = lqr(A_estim, B_estim, Q, R);

cutoff = 3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation parameters
dt = 0.01; % Time step (seconds)
total_time = 1000; % Total simulation time (seconds)
num_steps = total_time / dt; % Number of simulation steps

% Initialize state vectors
x = zeros(6, num_steps); % State vector [position; velocity]
u = zeros(2, num_steps); % Control input vector
estim_x = zeros(4, num_steps);
virtual_state = zeros(4, num_steps);
virtual_u = zeros(2, num_steps);
um = zeros(1, num_steps);
filtered_um = zeros(1, num_steps);

time = 0:dt:(total_time-dt); % Time vector

% Initial condition
x(:, 1) = [0; 0; 0; 0; 0; 0]; % Initial position and velocity
estim_x(:, 1) = [l; 0; 0; 0];

B_usv = [1/M(1,1), 1/M(1,1);
         0, 0;
         l/M(3,3), -l/M(3,3)
         0, 0;
         0, 0;
         0, 0]; 

B_disturbance = [0, 0;
                 0, 0;
                 0, 0;
                 1, 0;
                 0, 1;
                 0, 0]; 


% Simulation loop (Euler method)
for k = 2:num_steps-1
    

    % Compute the state derivative with disturbance
    f_usv = [(Xu*x(1, k) + m22*x(2, k)*x(3, k))/m11;
          (Yv*x(2, k) + Yr*x(3, k) - m11*x(1, k)*x(2, k))/m22;
          (-m22*x(1, k)*x(2, k) + m11*x(1, k)*x(1, k) + Nv*x(2, k) + Nr*x(3, k))/m33;
          x(1, k)*cos(x(6, k)) - x(2, k)*sin(x(6, k));
          x(1, k)*sin(x(6, k)) + x(2, k)*cos(x(6, k));
          x(3, k)];


    % L1 estim
    virtual_state(:, k) = [x(4, k) + l*cos(x(6, k));
                     x(5, k) + l*sin(x(6, k));
                     x(1, k)*cos(x(6, k)) - x(2, k)*sin(x(6, k)) - x(3, k)*l*sin(x(6, k));
                     x(1, k)*sin(x(6, k)) + x(2, k)*cos(x(6, k)) + x(3, k)*l*cos(x(6, k))];
    
    x_error = estim_x(:, k) - virtual_state(:,k);

    gain = -1;
    pi = (1/gain)*(exp(gain*dt)-1);
    um(k) = -exp(gain*dt)*x_error(3)/pi;
    filtered_um(k) = filtered_um(k-1)*exp(-cutoff*dt) - um(k)*(1-exp(-cutoff*dt));

    virtual_u(:,k) = [f_usv(1)*cos(x(6, k)) - x(1, k)*x(3, k)*sin(x(6, k)) - f_usv(2)*sin(x(6, k)) - x(2, k)*x(3, k)*cos(x(6, k)) - f_usv(3)*l*sin(x(6, k)) - x(3, k)*x(3, k)*l*cos(x(6, k));
                 f_usv(1)*sin(x(6, k)) + x(1, k)*x(3, k)*cos(x(6, k)) + f_usv(2)*sin(x(6, k)) - x(2, k)*x(3, k)*sin(x(6, k)) + f_usv(3)*l*cos(x(6, k)) - x(3, k)*x(3, k)*l*sin(x(6, k))];

    state_feedback = -K*virtual_state(:,k);
    state_feedback_u = [0.5*(m33*(state_feedback(2)*cos(x(6, k)) - state_feedback(1)*sin(x(6, k)))/(l*l) + m11*(state_feedback(1)*cos(x(6, k)) + state_feedback(2)*sin(x(6, k))));
                       0.5*(-m33*(state_feedback(2)*cos(x(6, k)) - state_feedback(1)*sin(x(6, k)))/(l*l) + m11*(state_feedback(1)*cos(x(6, k)) + state_feedback(2)*sin(x(6, k))))];

    % u(:,k) = -K*virtual_state(:,k) - um(k);
    u(:,k) = [0.5*(m11*cos(x(6, k))-m33*sin(x(6, k))/(l*l));
              0.5*(m11*cos(x(6, k))+m33*sin(x(6, k))/(l*l))]*filtered_um(k) + state_feedback_u;
    
 
    % Update the state using Euler method
    % estim_dx = A_estim*estim_x(:,k) + B_estim*virtual_u(:,k) + B_estim*[um(k) ; 0] - L*(x_error);
    % dx = f_usv + B_usv*u(:, k) + B_usv*[0.5;0];

    Am = A_estim - B_estim*K;
    estim_dx = Am*estim_x(:,k) + B_estim*virtual_u(:,k) + B_estim*[um(k) ; 0] - L*(x_error);

    disturbance = [100*sin(10*time(k))+ 500; 0.0];
    dx = f_usv + B_usv*u(:, k) + B_usv*disturbance;

    x(:, k+1) = x(:, k) + dx * dt;
    estim_x(:, k+1) = estim_x(:, k) + estim_dx * dt;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
subplot(3,1,1);
plot(time, x(1, :));
title('Position real vs Time');
xlabel('Time (s)');
ylabel('Position real');

subplot(3,1,2);
plot(time, um);
title('Disturbance vs Time');
xlabel('Time (s)');
ylabel('Disturbance');

subplot(3,1,3);
plot(time, u);
title('Control vs Time');
xlabel('Time (s)');
ylabel('Control');