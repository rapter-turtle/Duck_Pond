clear

% A_estim = [0, 0, 1, 0;
%      0, 0, 0, 1;
%      0, 0, 0, 0;
%      0, 0, 0, 0];
% 
% B_estim = [0, 0;
%      0, 0;
%      1, 0;
%      0, 1];
% 
% L = diag([1, 1, 1, 1]);
% 
% % Define the state weighting matrix Q and control weighting matrix R
% Q = diag([1, 1, 1, 1]); % Adjust the values based on your requirements
% R = diag([1, 1]);          % Adjust the values based on your requirements
% 
% % Calculate the LQR gain K
% [K, S, e] = lqr(A_estim, B_estim, Q, R);
% 
% K1 = [1, 0, 1.73, 0;0,1,0,1.73];
% Am = A_estim-B_estim*K1
% eig(A_estim-B_estim*K1)

syms s

% Define the matrix A
A = [s, 0, -1, 0;
     0, s, 0, -1;
     1, 0, s+1.73, 0;
     0, 1, 0, s+1.73];

% Initialize the cofactor matrix
cof_A = sym(zeros(4, 4));

% Compute the cofactor matrix
for i = 1:4
    for j = 1:4
        M_ij = A;
        M_ij(i,:) = [];  % Remove i-th row
        M_ij(:,j) = [];  % Remove j-th column
        cof_A(i, j) = (-1)^(i+j) * det(M_ij); % Calculate cofactor
    end
end

% Calculate the adjugate matrix by transposing the cofactor matrix
adj_A = transpose(cof_A);

% Display the adjugate matrix
disp('Adjugate of A:');
disp(adj_A);

C = [1 0 0 0;0 1 0 0];
% C = [1 0 0 0];
Bm = [0 0 ;0 0;1 0; 0 1];
Bum = [1 0;0 1;0 0; 0 0];

Hm = C*adj_A*Bm;
Hum = C*adj_A*Bum;

H = inv(Hm)*Hum