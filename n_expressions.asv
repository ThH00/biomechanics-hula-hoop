syms psi theta phi real


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

syms beta real
n = cos(beta)*e1+sin(beta)*e2

n =
 
cos(beta)*(cos(phi)*cos(psi) - cos(theta)*sin(phi)*sin(psi)) - sin(beta)*(cos(psi)*sin(phi) + cos(phi)*cos(theta)*sin(psi))
cos(beta)*(cos(phi)*sin(psi) + cos(psi)*cos(theta)*sin(phi)) - sin(beta)*(sin(phi)*sin(psi) - cos(phi)*cos(psi)*cos(theta))
                                                              cos(beta)*sin(phi)*sin(theta) + cos(phi)*sin(beta)*sin(theta)
 
diff = simplify(n(1)^2+n(2)^2)

replace 
psi, phi, theta and plot dunction of beta
