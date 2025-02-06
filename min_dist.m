% sweep angle
syms gamma real

E1 = [1;0;0];
E2 = [0;1;0];
E3 = [0;0;1];

%% Hip
% the coordinares A1, A2, A2 are given
% the function a(z) is given

% coordinates of the bottom axis of the hip
syms A1 A2 A3 real
xA = [A1, A2, A3];

syms z real
syms a(z) real

% position vector of point on hip
xhip = xA+a(z)*cos(gamma)*E1+a(z)*sin(gamma)*E2;

%% Hoop
% the dimension b is given
% the generalized alpha code outputs B1, B2, B3, psi, theta, phi

% radius of hoop
syms b real

% coordinates of hoop center
syms B1 B2 B3 real      
xB = [B1, B2, B3];

% euler angles of hoop
syms psi theta phi real

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

% position vector of point on hoop
xhoop = xB+b*(cos(gamma)*e1+sin(gamma)*e2);


% symvar(xhoop) = [B1, B2, B3, b, gamma, phi, psi, theta]

%% Min distance between hip and hoop
% square of the min distance between hip and hoop
d2 = dot(xhoop-xhip,xhoop-xhip);

% symvar(d_squared) = [A1, A2, A3, B1, B2, B3, b, gamma, phi, psi, theta, z]
% also a(z) is a given function
% We want to find z and gamma to minimize d_squared

d2_dz = diff(d2,z)




