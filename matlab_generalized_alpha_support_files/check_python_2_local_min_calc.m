% testing python's calculation of two local minima

tol_n = 1.0e-6;
n_tau = 1/tol_n;

% Hoop position coordinates
R_hoop = 0.5;
xbar_hoop = [0.12380409, 0.74473118, 0.37049737];
psi = 0.65189912;
theta = 0.26993748;
phi = 0.34501345;

% Hip position coordinates
R_hip = 0.2;
xbar_hip = [-0.09043523,  0.53515719,  0.        ];

% Fixed basis
E1 = [1;0;0];
E2 = [0;1;0];
E3 = [0;0;1];

% Rotation matrices
R1 = [cos(psi), sin(psi), 0;
      -sin(psi), cos(psi), 0;
      0, 0, 1];
  
R2 = [1, 0, 0;
      0, cos(theta), sin(theta);
      0, -sin(theta), cos(theta)];

R3 = [cos(phi), sin(phi), 0;
      -sin(phi), cos(phi), 0;
      0, 0, 1];

% Euler basis
% {E1, E2, E3} components of {e1',e2',e3'}
e1p = R1'*E1;
e2p = R1'*E2;
e3p = R1'*E3;

% {E1, E2, E3} components of {e1'',e2'',e3''}
e1pp = (R2*R1)'*E1;
e2pp = (R2*R1)'*E2;
e3pp = (R2*R1)'*E3;

% {E1, E2, E3} components of {e1,e2,e3}
e1 = (R3*R2*R1)'*E1;
e2 = (R3*R2*R1)'*E2;
e3 = (R3*R2*R1)'*E3;


% Create an array of possible tau values (step size < algorithm tolerance)
tau = linspace(0, 2*pi, n_tau);
% tau = tau(1:end-1); % removing last point to avoid repetition
% I can find intervals containing the minima and then refine the discretization in these intervals (or use the bisection method)

% Creating array of hoop points
% Reshape tau to (1000000, 1) to enable broadcasting
u = cos(tau)'*e1' + sin(tau)'*e2';  % Shape (1000000, 3)

xM = xbar_hoop+R_hoop*u;

% Calculating the value of dH for each point
dv = zeros(n_tau-1,1);
dh = zeros(n_tau-1,1);
for i = 1:(n_tau-1)
    dv(i) = dot(xM(i,:),E3');
    temp = xM(i,:)-dv(i)*E3'-xbar_hip;
    % Compute the norm of each row
    dh(i) = norm(temp);
end

% Find the minimizers of dh
% Find local minima (less than neighbors)
min_indices = islocalmin(dh);
% Find the minizing value of tau
minimizing_tau = tau(min_indices);


 