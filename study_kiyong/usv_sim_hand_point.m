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
cutoff = 3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation parameters
dt = 0.01; % Time step (seconds)
total_time = 100; % Total simulation time (seconds)
num_steps = total_time / dt; % Number of simulation steps

% Initialize state vectors
x = zeros(6, num_steps); % State vector [position; velocity]
x_abs = zeros(6, num_steps);
u = zeros(2, num_steps);
matched_u = zeros(2, num_steps); % Control input vector
unmatched_u = zeros(2, num_steps);
smc_list = zeros(2, num_steps);
disturbance_list = zeros(4, num_steps);
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
x(:, 1) = [0; 0; 0; -l; 20; 0]; % Initial position and velocity
x(:, 2) = x(:, 1);
x_abs(:, 1) = x(:, 1); % Initial position and velocity
x_abs(:, 2) = x_abs(:, 1);
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

s1 = 0;
s2 = 0;

% Simulation loop (Euler method)
for k = 2:num_steps-1
    % Disturbance
    xy_dis = [500*sin(0.5*dt*k);500*cos(0.5*dt*k);200*cos(0.1*dt*k)/l];
    disturbance = [0; 0; xy_dis(1)*cos(x(6,k)) + xy_dis(2)*sin(x(6,k)); -xy_dis(1)*sin(x(6,k)) + xy_dis(2)*cos(x(6,k)); xy_dis(3)];
    disturbance_list(:,k) = [disturbance(1:2) ; xy_dis(1) - xy_dis(3)*sin(x(6,k)); xy_dis(2) + xy_dis(3)*cos(x(6,k))];

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
                     x(1, k)*cos(x(6, k)) - x(2, k)*sin(x(6, k)) - x(3, k)*l*sin(x(6, k)) + disturbance(1);
                     x(1, k)*sin(x(6, k)) + x(2, k)*cos(x(6, k)) + x(3, k)*l*cos(x(6, k)) + disturbance(2)];

    s1 = virtual_state(1, k) + virtual_state(2, k) + virtual_state(3, k) + virtual_state(4, k);
    s2 = x(6,k) + x(3,k);

    tau_r = -x(3,k) + f_usv(2)/l - 100*sat(s2,-1,1) - virtual_state(3, k)*virtual_state(3, k) - virtual_state(4, k)*virtual_state(4, k) ;
    tau_u = 0;%(-tau_r*(-l*sin(x(6,k)+l*cos(x(6,k)))) - virtual_state(3,k) - virtual_state(4,k) - 10*sat(s1,-1,1))/(cos(x(6,k)) + sin(x(6,k)));;


    % Virtual control
    x_abs(:,k) = [x(1,k) + disturbance(1)*cos(x(6,k)) + disturbance(2)*sin(x(6,k));
                  x(2,k) - disturbance(1)*sin(x(6,k)) + disturbance(2)*cos(x(6,k));
                  x(3,k);
                  x(4,k);
                  x(5,k);
                  x(6,k)];

    f_usv_virtual = [(Xu*x_abs(1, k))/m11;
          (Yv*x_abs(2, k) + Yr*x_abs(3, k))/m22;
          (Nv*x_abs(2, k) + Nr*x_abs(3, k))/m33;
          x_abs(1, k)*cos(x(6, k)) - x_abs(2, k)*sin(x(6, k));
          x_abs(1, k)*sin(x(6, k)) + x_abs(2, k)*cos(x(6, k));
          x(3, k)];

    virtual_u(:,k) = [f_usv_virtual(1)*cos(x(6, k)) - x(1, k)*x(3, k)*sin(x(6, k)) - f_usv_virtual(2)*sin(x(6, k)) - x(2, k)*x(3, k)*cos(x(6, k)) - f_usv_virtual(3)*l*sin(x(6, k)) - x(3, k)*x(3, k)*l*cos(x(6, k));
                 f_usv_virtual(1)*sin(x(6, k)) + x(1, k)*x(3, k)*cos(x(6, k)) + f_usv_virtual(2)*sin(x(6, k)) - x(2, k)*x(3, k)*sin(x(6, k)) + f_usv_virtual(3)*l*cos(x(6, k)) - x(3, k)*x(3, k)*l*sin(x(6, k))];

    smc = [tau_u*cos(x(6,k)) - l*tau_r*sin(x(6,k));tau_u*sin(x(6,k)) + l*tau_r*cos(x(6,k))];
    smc_list(:,k) = smc;
    % Real Control input
    u(:,k) = [0.5*(m11*cos(x(6, k)) - m33*sin(x(6, k))/(l*l)), 0.5*(m11*sin(x(6, k))+m33*cos(x(6, k))/(l*l));
              0.5*(m11*cos(x(6, k)) + m33*sin(x(6, k))/(l*l)), 0.5*(m11*sin(x(6, k))-m33*cos(x(6, k))/(l*l))]*(- virtual_u(:,k) + smc);% + state_feedback_u;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if u(1,k) < -2000
        u(1,k) = -2000;
    end
    if u(2,k) < -2000
        u(2,k) = -2000;
    end    
    dx = f_usv + B_usv*u(:, k) + Bm_disturbance*disturbance(3:5) + Bum_disturbance*disturbance(1:2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %System update
    x(:, k+1) = x(:, k) + dx * dt;
    % estim_x(:, k+1) = estim_x(:, k) + estim_dx * dt;

    pi = 3.1415926535;
    if x(6,k+1) > pi
        x(6,k+1) = x(6,k+1) - 2*pi;
    elseif x(6,k+1) < -pi
        x(6,k+1) = x(6,k+1) + 2*pi;
    end    

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fontSize = 15;
axisfontSize = 12;
fontSize_T = 18;

figure;
subplot(3,2,1);
plot(time, virtual_state(1,:), 'DisplayName', 'Estimated'); % Label for the first line
title('Head Point Position X', 'FontSize', fontSize_T);
xlabel('Time(s)', 'FontSize', fontSize);
ylabel('X(m)', 'FontSize', fontSize);
% ylim([0, 5]);
set(gca, 'FontSize', axisfontSize);


subplot(3,2,2);
plot(time, virtual_state(2,:), 'DisplayName', 'Estimated'); % Label for the first line
title('Head Point Position Y', 'FontSize', fontSize_T);
xlabel('Time(s)', 'FontSize', fontSize);
ylabel('Y(m)', 'FontSize', fontSize);
% ylim([0, 5]);
set(gca, 'FontSize', axisfontSize);

% 
subplot(3,2,3);
plot(time, smc_list(1,:), 'DisplayName', 'Estimated'); % Label for the first line
hold on;
title('X_con', 'FontSize', fontSize_T);
xlabel('Time(s)', 'FontSize', fontSize);
ylabel('tau_x', 'FontSize', fontSize);
% ylim([0, 2]);
legend('show'); % Display the legend with labels
set(gca, 'FontSize', axisfontSize);


subplot(3,2,4);
plot(time, smc_list(2,:), 'DisplayName', 'Estimated'); % Label for the first line
hold on;
title('Y_con', 'FontSize', fontSize_T);
xlabel('Time(s)', 'FontSize', fontSize);
ylabel('tau_y', 'FontSize', fontSize);
% ylim([0, 2]);
legend('show'); % Display the legend with labels
set(gca, 'FontSize', axisfontSize);


subplot(3,2,5);
plot(time, x(6,:), 'DisplayName', 'Estimated'); % Label for the first line
hold on;
title('Heading', 'FontSize', fontSize_T);
xlabel('Time(s)', 'FontSize', fontSize);
ylabel('Heading', 'FontSize', fontSize);
% ylim([-800, 800]);
legend('show'); % Display the legend with labels
set(gca, 'FontSize', axisfontSize);

% 
% subplot(3,2,6);
% plot(time, m11*um(4,:), 'DisplayName', 'Estimated'); % Label for the first line
% hold on;
% plot(time, disturbance_list(4,:), 'DisplayName', 'Ground truth'); % Label for the second line
% hold off;
% title('Matched Uncertainty : Y Wave Force', 'FontSize', fontSize_T);
% xlabel('Time(s)', 'FontSize', fontSize);
% ylabel('Force(N)', 'FontSize', fontSize);
% ylim([-800, 800]);
% legend('show'); % Display the legend with labels
% set(gca, 'FontSize', axisfontSize);

% 
% subplot(4,2,7);
% plot(time, u(1,:), 'DisplayName', 'Thruster L'); % Label for the first line
% title('Thruster Control Input Left', 'FontSize', fontSize_T);
% xlabel('Time(s)', 'FontSize', fontSize);
% ylabel('Force(N)', 'FontSize', fontSize);
% % ylim([0, 3]);
% legend('show'); % Display the legend with labels
% set(gca, 'FontSize', axisfontSize);
% 
% 
% subplot(4,2,8);
% plot(time, u(2,:), 'DisplayName', 'Thruster R'); % Label for the first line
% title('Thruster Control Input Right', 'FontSize', fontSize_T);
% xlabel('Time(s)', 'FontSize', fontSize);
% ylabel('Force(N)', 'FontSize', fontSize);
% % ylim([0, 3]);
% legend('show'); % Display the legend with labels
% set(gca, 'FontSize', axisfontSize);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the ship polygon
ship_length = 7; % Length of the ship
ship_width = 3; % Width of the ship
ship_shape = [-ship_length/2 + ship_width, -ship_width/2; 
              ship_length/2, 0; 
              -ship_length/2 + ship_width, ship_width/2; 
              -ship_length/2 - ship_width, ship_width/2; 
              -ship_length/2 - ship_width, -ship_width/2];

% Create a figure for animation
figure;
hold on;
axis equal;
xlabel('X Position');
ylabel('Y Position');
title('No control');
xlim([-30, 30]); % Set the X-axis limits
ylim([-30, 30]); % Set the Y-axis limits
grid on;
legend('Own ship');


% Set up the video writer
video = VideoWriter('L1.avi'); % Create a video writer object
video.FrameRate = 30; % Set the frame rate
open(video); % Open the video file

% Animation loop
for k = 1:10:num_steps
    % Compute ship's current position and orientation
    X = x(4, k);
    Y = x(5, k);
    psi = x(6, k);

    % Rotate and translate the ship shape
    R = [cos(psi), -sin(psi); sin(psi), cos(psi)];
    ship_position = (R * ship_shape')';
    ship_position(:, 1) = ship_position(:, 1) + X;
    ship_position(:, 2) = ship_position(:, 2) + Y;

    % Plot the ship
    ship_plot = fill(ship_position(:, 1), ship_position(:, 2), 'r', 'FaceAlpha', 0.5, 'DisplayName', 'Own Ship');

    % Capture the current frame
    frame = getframe(gcf);
    writeVideo(video, frame);

    % Pause to create animation effect
    pause(0.01);

    % Remove the current plot to update the next frame
    if k < num_steps
        delete(ship_plot);
    end
end

hold off;

% Close the video file
close(video);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Stamp trace %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Define the ship polygon
% ship_length = 7; % Length of the ship
% ship_width = 3; % Width of the ship
% ship_shape = [-ship_length/2 + ship_width, -ship_width/2; 
%               ship_length/2, 0; 
%               -ship_length/2 + ship_width, ship_width/2; 
%               -ship_length/2 - ship_width, ship_width/2; 
%               -ship_length/2 - ship_width, -ship_width/2];
% 
% % Create a figure for the plot
% figure;
% hold on;
% axis equal;
% xlabel('X Position');
% ylabel('Y Position');
% title('USV Position Stamps');
% xlim([-30, 30]); % Set the X-axis limits
% ylim([-30, 30]); % Set the Y-axis limits
% grid on;
% 
% 
% 
% % Plot USV position stamps every 10 seconds
% for k = 1:num_steps
%     if k == 1 || mod(k, 3/dt) == 0
%         % Compute ship's current position and orientation
%         X = x(4, k);
%         Y = x(5, k);
%         psi = x(6, k);
% 
%         % Rotate and translate the ship shape
%         R = [cos(psi), -sin(psi); sin(psi), cos(psi)];
%         ship_position = (R * ship_shape')';
%         ship_position(:, 1) = ship_position(:, 1) + X;
%         ship_position(:, 2) = ship_position(:, 2) + Y;
% 
%         % Plot the ship
%         fill(ship_position(:, 1), ship_position(:, 2), 'b', 'FaceAlpha', 0.5, 'DisplayName', 'Own Ship');
%     end
% end
% 
% % Add a legend after the loop to ensure it doesn't duplicate entries
% legend('USV Position');
% 
% hold off;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function y = sat(x, lower_limit, upper_limit)
    % Saturation function limits x between lower_limit and upper_limit
    y = max(min(x, upper_limit), lower_limit);
end
