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

Bum_estim = [1, 0;
     0, 1;
     0, 0;
     0, 0];

L = diag([1, 1, 1, 1]);
K = [1, 0, 1.73, 0;0,1,0,1.73];
cutoff = 4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation parameters
dt = 0.01; % Time step (seconds)
total_time = 100; % Total simulation time (seconds)
num_steps = total_time / dt; % Number of simulation steps

% Initialize state vectors
x = zeros(6, num_steps); % State vector [position; velocity]
u = zeros(2, num_steps);
matched_u = zeros(2, num_steps); % Control input vector
unmatched_u = zeros(2, num_steps);
statefeedback_u = zeros(2, num_steps);

estim_x = zeros(4, num_steps);
virtual_state = zeros(4, num_steps);
virtual_u = zeros(2, num_steps);
um = zeros(4, num_steps);
um_dot = 0;
con = [0;0;0;0];
filtered_um = zeros(4, num_steps);

time = 0:dt:(total_time-dt); % Time vector

% Initial condition
x(:, 1) = [0; 0; 0; -l; 0; 0]; % Initial position and velocity
x(:, 2) = x(:, 1);
estim_x(:, 1) = [x(4,1)+l; 0; 0; 0];
estim_x(:, 2) = estim_x(:, 1);

B_usv = [1/M(1,1), 1/M(1,1);
         0, 0;
         l/M(3,3), -l/M(3,3)
         0, 0;
         0, 0;
         0, 0]; 

Bm_disturbance = [1/M(1,1), 0, 0;
                 0, 1/M(1,1), 0;
                 0, 0, l/M(3,3);
                 0, 0, 0;
                 0, 0, 0;
                 0, 0, 0]; 

Bum_disturbance = [0, 0;
                 0, 0;
                 0, 0;
                 1, 0;
                 0, 1;
                 0, 0]; 


% Simulation loop (Euler method)
for k = 2:num_steps-1
    

    % Compute the state derivative with disturbance
    f_usv = [(Xu*x(1, k))/m11;
          (Yv*x(2, k) + Yr*x(3, k))/m22;
          (Nv*x(2, k) + Nr*x(3, k))/m33;
          x(1, k)*cos(x(6, k)) - x(2, k)*sin(x(6, k));
          x(1, k)*sin(x(6, k)) + x(2, k)*cos(x(6, k));
          x(3, k)];


    % L1 estim
    virtual_state(:, k) = [x(4, k) + l*cos(x(6, k));
                     x(5, k) + l*sin(x(6, k));
                     x(1, k)*cos(x(6, k)) - x(2, k)*sin(x(6, k)) - x(3, k)*l*sin(x(6, k));
                     x(1, k)*sin(x(6, k)) + x(2, k)*cos(x(6, k)) + x(3, k)*l*cos(x(6, k))];
    % L1 control input
    x_error = estim_x(:, k) - virtual_state(:,k);
    gain = -1;
    pi = (1/gain)*(exp(gain*dt)-1);
    um(:,k) = -exp(gain*dt)*x_error/pi;
    um_dot = (um(:,k)-um(:,k-1))/dt;
    % matched uncertainty
    con(3:4) = um(3:4,k);
    % unmatched uncertainty
    con(1:2) = um_dot(1:2);
    % Low pass filtering
    filtered_um(:,k) = filtered_um(:,k-1)*exp(-cutoff*dt) - con*(1-exp(-cutoff*dt));
    matched_u(:,k) = filtered_um(3:4,k);
    unmatched_u(:,k) = filtered_um(1:2,k);

    % Nominal control
    % nominal_control = [0;0];
    % estim_nominal_control = [(nominal_control(1) + nominal_control(2))*cos(x(6, k))/m11 - (nominal_control(1)-nominal_control(2))*l*l*sin(x(6, k))/m33;
    %                          (nominal_control(1) + nominal_control(2))*sin(x(6, k))/m11 + (nominal_control(1)-nominal_control(2))*l*l*cos(x(6, k))/m33];

    % Virtual control
    virtual_u(:,k) = [f_usv(1)*cos(x(6, k)) - x(1, k)*x(3, k)*sin(x(6, k)) - f_usv(2)*sin(x(6, k)) - x(2, k)*x(3, k)*cos(x(6, k)) - f_usv(3)*l*sin(x(6, k)) - x(3, k)*x(3, k)*l*cos(x(6, k));
                 f_usv(1)*sin(x(6, k)) + x(1, k)*x(3, k)*cos(x(6, k)) + f_usv(2)*sin(x(6, k)) - x(2, k)*x(3, k)*sin(x(6, k)) + f_usv(3)*l*cos(x(6, k)) - x(3, k)*x(3, k)*l*sin(x(6, k))];
    % State feedback for Hurwitz matrix
    state_feedback = -K*virtual_state(:,k);
    state_feedback_u = [0.5*(m33*(state_feedback(2)*cos(x(6, k)) - state_feedback(1)*sin(x(6, k)))/(l*l) + m11*(state_feedback(1)*cos(x(6, k)) + state_feedback(2)*sin(x(6, k))));
                       0.5*(-m33*(state_feedback(2)*cos(x(6, k)) - state_feedback(1)*sin(x(6, k)))/(l*l) + m11*(state_feedback(1)*cos(x(6, k)) + state_feedback(2)*sin(x(6, k))))];
    statefeedback_u(:,k) = state_feedback_u;

    % unmatched_u(:,k) = [0;0];
    matched_u(:,k) = [0;0];
    % Real Control input
    u(:,k) = [0.5*(m11*cos(x(6, k)) - m33*sin(x(6, k))/(l*l)), 0.5*(m11*sin(x(6, k))+m33*cos(x(6, k))/(l*l));
              0.5*(m11*cos(x(6, k)) + m33*sin(x(6, k))/(l*l)), 0.5*(m11*sin(x(6, k))-m33*cos(x(6, k))/(l*l))]*(matched_u(:,k) + unmatched_u(:,k) - virtual_u(:,k));% + state_feedback_u;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    estim_dx = (A_estim)*estim_x(:,k) + B_estim*(matched_u(:,k) + unmatched_u(:,k)) + B_estim*um(3:4,k) + Bum_estim*um(1:2,k) - L*(x_error);
    % estim_dx = (A_estim)*estim_x(:,k) + B_estim*virtual_u(:,k) + B_estim*(matched_u(:,k) + unmatched_u(:,k)) + B_estim*um(3:4,k) + Bum_estim*um(1:2,k) - L*(x_error);

    xy_dis = [500*sin(0.5*dt*k);500*sin(0.5*dt*k)];
    disturbance = [1; 2; xy_dis(1)*cos(x(6,k)) + xy_dis(2)*sin(x(6,k)); -xy_dis(1)*sin(x(6,k)) + xy_dis(2)*cos(x(6,k)); 0];
    
    dx = f_usv + B_usv*u(:, k) + Bm_disturbance*disturbance(3:5) + Bum_disturbance*disturbance(1:2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    %System update
    x(:, k+1) = x(:, k) + dx * dt;
    estim_x(:, k+1) = estim_x(:, k) + estim_dx * dt;

    pi = 3.1415926535;
    if x(6,k+1) > pi
        x(6,k+1) = x(6,k+1) - 2*pi;
    elseif x(6,k+1) < -pi
        x(6,k+1) = x(6,k+1) + 2*pi;
    end    

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
subplot(4,1,1);
plot(time, virtual_state(1,:));
title('Headpoint x');
xlabel('Time (s)');
ylabel('Headpoint x');

subplot(4,1,2);
plot(time, virtual_state(2,:));
title('Headpoint y');
xlabel('Time (s)');
ylabel('Headpoint y');

subplot(4,1,3);
plot(time,um(1:2,:));
title('vel dis');
xlabel('Time (s)');
ylabel('vel dis');

subplot(4,1,4);
plot(time,[um(3,:); -filtered_um(3,:)]);
title('Estimated disturbance 3, 4');
xlabel('Time (s)');
ylabel('Estimated disturbance');
