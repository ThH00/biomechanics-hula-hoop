% E1 = [1;0;0];
% E2 = [0;1;0];
% E3 = [0;0;1];

syms E1 E2 E3 real
syms psi theta phi real

e1p = cos(psi)*E1+sin(psi)*E2;
e2p = -sin(psi)*E1+cos(psi)*E2;
e3p = E3;

e1pp = e1p;
e2pp = cos(theta)*e2p+sin(theta)*e3p;
e3pp = -sin(theta)*e2p+cos(theta)*e3p;

e1 = cos(phi)*e1pp+sin(phi)*e2pp;
e2 = -sin(phi)*e1pp+cos(phi)*e2pp;
e3 = e3pp;

de1_dpsi = diff(e1,psi)
de2_dpsi = diff(e2,psi)
de1_dtheta = diff(e1,theta)
de2_dtheta = diff(e2,theta)
de1_dphi = diff(e1,phi)
de2_dphi = diff(e2,phi)