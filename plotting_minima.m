% Hoop position coordinates
R_hoop = 0.5;
xbar_hoop = [0.1, 0, 1];
psi = 0;
theta = pi/6;
phi = 0;

% Hip position coordinates
R_hip = 0.2;
xbar_hip = [0.2, 0, 0];

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

% Plot
figure()
hold on
xlim([-2, 2])
ylim([-2, 2])
zlim([-2, 2])
view(45, 45)
axis square
box on

m = 100;    % hoop refinement
tau = linspace(0,2*pi,m);

% plotting the hip
z = linspace(-1.5,1.5,m);
r = 0.2*ones(m,1);
for j = 1:2:length(tau)
    hip(j) = plot3(xbar_hip(1)+r*cos(tau(j)),xbar_hip(2)+r*sin(tau(j)),z,'k');
end

% plotting the hoop
circle = plot3(xbar_hoop(1)+R_hoop*cos(tau)*e1(1)+R_hoop*sin(tau)*e2(1), ...
    xbar_hoop(2)+R_hoop*cos(tau)*e1(2)+R_hoop*sin(tau)*e2(2), ...
    xbar_hoop(3)+R_hoop*cos(tau)*e1(3)+R_hoop*sin(tau)*e2(3), ...
    'color','b','LineWidth',2);

center_plot = plot3(xbar_hoop(1), xbar_hoop(2), xbar_hoop(3), 'b','Marker', '*', 'LineWidth',2);

e1_plot = quiver3(xbar_hoop(1), xbar_hoop(2), xbar_hoop(3), e1(1), e1(2), e1(3), 'r');
e2_plot = quiver3(xbar_hoop(1), xbar_hoop(2), xbar_hoop(3), e2(1), e2(2), e2(3), 'g');
e3_plot = quiver3(xbar_hoop(1), xbar_hoop(2), xbar_hoop(3), e3(1), e3(2), e3(3), 'b');

% plotting the minima
% first
x3_1 = 0.995495631367135;
tau_1 = -3.11155901441972;
n_1 = [cos(tau_1)*e1(1)+sin(tau_1)*e2(1),...
    cos(tau_1)*e1(2)+sin(tau_1)*e2(2),...
    cos(tau_1)*e1(3)+sin(tau_1)*e2(3)];
min_hoop_1 = plot3(xbar_hoop(1)+R_hoop*n_1(1), xbar_hoop(2)+R_hoop*n_1(2), xbar_hoop(3)+R_hoop*n_1(3), ...
    '*','color','r','LineWidth',2);
min_hip_1 = plot3(xbar_hip(1)+R_hip*n_1(1),xbar_hip(2)+R_hip*n_1(2),x3_1+R_hip*n_1(3), ...
    '*','color','r','LineWidth',2);


% second
x3_2 = 1.03098964904516;
tau_2 = 0.208096314839801;
n_2 = [cos(tau_2)*e1(1)+sin(tau_2)*e2(1),...
    cos(tau_2)*e1(2)+sin(tau_2)*e2(2),...
    cos(tau_2)*e1(3)+sin(tau_2)*e2(3)];
min_hoop_1 = plot3(xbar_hoop(1)+R_hoop*n_2(1), xbar_hoop(2)+R_hoop*n_2(2), xbar_hoop(3)+R_hoop*n_2(3), ...
    '*','color','g','LineWidth',2);
min_hip_1 = plot3(xbar_hip(1)+R_hip*n_2(1),xbar_hip(2)+R_hip*n_2(2),x3_1+R_hip*n_2(3), ...
    '*','color','g','LineWidth',2);


% normal direction along loop circumference
n = [cos(tau)*e1(1)+sin(tau)*e2(1);...
    cos(tau)*e1(2)+sin(tau)*e2(2);...
    cos(tau)*e1(3)+sin(tau)*e2(3)];

rel_vector = xbar_hoop'.*ones(3,100)+R_hoop*n-(xbar_hip'.*ones(3,100)+R_hip*n);
for i = 1:m
    distance(i) = norm(rel_vector(:,i));
end
