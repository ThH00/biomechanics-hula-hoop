folder_path = "/Users/theresahonein/Desktop/terryhonein/Research-HulaHoop/outputs/2025-07-01_21-20-37";

close all
clear all

load(folder_path+'/q.mat')
load(folder_path+'/u.mat')
load(folder_path+'/contacts.mat')

%% Heights
m = 0.2;
g = 1;
R_hoop = 0.5;
R_hip = 0.2;

E1 = [1;0;0];
E2 = [0;1;0];
E3 = [0;0;1];

ntime = 400;

x1 = squeeze(q(1,1,1:ntime));
x2 = squeeze(q(1,2,1:ntime));
x3 = squeeze(q(1,3,1:ntime));
psi = squeeze(q(1,4,1:ntime));
theta = squeeze(q(1,5,1:ntime));
phi = squeeze(q(1,6,1:ntime));

% height of the center of mass of the hula hoop
figure(1)
hold on
title('Plotting Local Minima','Interpreter','Latex')
plot(x3,'LineWidth',1,'Color','b')

ntau = 10*6;
% Create an array of possible tau values (step size < algorithm tolerance)
tau = linspace(0, 2*pi, ntau);
% I can find intervals containing the minima and then refine the discretization in these intervals (or use the bisection method)

u_vec = zeros(ntau,3);
xM = zeros(ntau,3);
dv = zeros(ntau,1);
dh = zeros(ntau,1);
temp = zeros(ntau,3);

minimizing_tau = zeros(ntime,2);
contact_point_height = zeros(ntime,2);
gN = zeros(ntime,2);
minimum_dh = zeros(ntime,2);

for j = 1:ntime

    R1 = [cos(psi(j)), sin(psi(j)), 0; -sin(psi(j)), cos(psi(j)), 0; 0, 0, 1];
    R2 = [1, 0, 0; 0, cos(theta(j)), sin(theta(j)); 0, -sin(theta(j)), cos(theta(j))];
    R3 = [cos(phi(j)), sin(phi(j)), 0; -sin(phi(j)), cos(phi(j)), 0; 0, 0, 1];
    
    e1 = (R3*R2*R1)'*E1;
    e2 = (R3*R2*R1)'*E2;

    for i = 1:ntau
    
        % Creating array of hoop points
        u_vec(i,:) = cos(tau(i)) * e1 + sin(tau(i)) * e2;
        
        xM(i,:) = [x1(j); x2(j); x3(j)]+R_hoop*u_vec(i,:)';
        
        % Calculating the value of dH for each point
        dv(i) = dot(xM(i,:),E3);
        temp(i,:) = xM(i,:)'-dv(i)*E3-[0;0;0];

        % Compute the norm of each row
        dh(i) = norm(temp(i,:));
       
    end

    % figure(10)
    % clf
    % plot(dh)
    % hold on
    % plot([0,ntau],[0.2,0.2])
    % axis([0,ntau,0,1])

    % Find the minimizers of dh
    % Find local minima (less than neighbors)
    min_indices = find(islocalmin(dh));
    temp2 = tau(min_indices);
    if length(temp2) == 2
        minimizing_tau(j,:) = temp2;
        contact_point_height(j,:) = dv(min_indices);
        gN(j,:) = dh(min_indices)-R_hip;
        minimum_dh(j,:) = dh(min_indices);
    end
    if length(temp2) == 1
        minimizing_tau(j,1) = temp2;
        contact_point_height(j,1) = dv(min_indices);
        gN(j,1) = dh(min_indices)-R_hip;
        minimum_dh(j,1) = dh(min_indices);
    end
    if length(temp2) == 0
        min_indices = [1];
        temp2 = tau(min_indices);
        minimizing_tau(j,1) = temp2;
        contact_point_height(j,1) = dv(min_indices);
        gN(j,1) = dh(min_indices)-R_hip;
        minimum_dh(j,1) = dh(min_indices);
    end

    % figure(3)
    % clf
    % hold on
    % plot3(x1(j)+R_hoop*u(:,1),x2(j)+R_hoop*u(:,2),x3(j)+R_hoop*u(:,3),'LineWidth',1,'Color','b')
    % plot3(x1(j)+R_hoop*u(min_indices,1),x2(j)+R_hoop*u(min_indices,2),x3(j)+R_hoop*u(min_indices,3),'*','LineWidth',1,'Color','r')
    % axis equal
    % xlim([-2,2])
    % ylim([-2,2])
    % zlim([-2,2])
    % view(45,10)

end

figure(1)
hold on
plot(contact_point_height(:,1),'.','LineWidth',1,'Color','k')

idx = find(contact_point_height(:,2));
arr = 1:ntime;

plot(arr(idx),contact_point_height(idx,2),'.', 'LineWidth',1,'Color','k')

% plot(x3+R_hoop,'--', 'LineWidth',0.5,'Color','b')
% plot(x3-R_hoop,'--', 'LineWidth',0.5,'Color','b')

touching_contact_idx1 = find(gN(:,1)<10^(-6));
touching_contact_idx2 = find(gN(:,2)<10^(-6));

A = squeeze(contacts(1,1:2,1:ntime));
Atouching_contact_idx1 = find(A(1,:)==1);
Atouching_contact_idx2 = find(A(2,:)==1);

B = squeeze(contacts(1,3:4,1:ntime));
Btouching_contact_idx1 = find(B(1,:)==1);
Btouching_contact_idx2 = find(B(2,:)==1);

C = squeeze(contacts(1,5:6,1:ntime));
Ctouching_contact_idx1 = find(C(1,:)==1);
Ctouching_contact_idx2 = find(C(2,:)==1);

D = squeeze(contacts(1,7:8,1:ntime));
Dtouching_contact_idx1 = find(D(1,:)==1);
Dtouching_contact_idx2 = find(D(2,:)==1);

E = squeeze(contacts(1,9:10,1:ntime));
Etouching_contact_idx1 = find(E(1,:)==1);
Etouching_contact_idx2 = find(E(2,:)==1);

plot(arr(Atouching_contact_idx2),contact_point_height(Atouching_contact_idx2,2),'.', 'LineWidth',1,'Color','r')
plot(arr(Atouching_contact_idx1),contact_point_height(Atouching_contact_idx1,1),'.', 'LineWidth',1,'Color','r')

legend('Height of mass center','Height of potential contact point','Height of second contact point', 'Contact is active', 'Interpreter', 'latex');

figure(5)
hold on
% hip boundaries
plot([1,ntime],[0.2,0.2], 'LineWidth',1,'Color','g')
plot([1,ntime],[0,0], 'LineWidth',1,'Color','g')

plot(minimum_dh(:,1),'.','LineWidth',1,'Color','k')
plot(arr(idx),minimum_dh(idx,2),'.', 'LineWidth',1,'Color','k')

figure(4)
hold on
plot(gN(:,1),'LineWidth',1,'Color','k')
plot(gN(:,2),'LineWidth',1,'Color','b')



%% Energy

u1 = squeeze(u(1,1,1:ntime));
u2 = squeeze(u(1,2,1:ntime));
u3 = squeeze(u(1,3,1:ntime));
psidot = squeeze(u(1,4,1:ntime));
thetadot = squeeze(u(1,5,1:ntime));
phidot = squeeze(u(1,6,1:ntime));

I = [0.5*m*R_hoop^2, 0,0;
    0,0.5*m*R_hoop^2, 0;
    0,0,m*R_hoop^2];

T = zeros(ntime,1);
E = zeros(ntime,1);

for j = 1:ntime

    R1 = [cos(psi(j)), sin(psi(j)), 0; -sin(psi(j)), cos(psi(j)), 0; 0, 0, 1];
    R2 = [1, 0, 0; 0, cos(theta(j)), sin(theta(j)); 0, -sin(theta(j)), cos(theta(j))];
    R3 = [cos(phi(j)), sin(phi(j)), 0; -sin(phi(j)), cos(phi(j)), 0; 0, 0, 1];
    
    e1p = R1'*E1;
    e1 = (R3*R2*R1)'*E1;
    e2 = (R3*R2*R1)'*E2;
    e3 = (R3*R2*R1)'*E3;
    
    omega = psidot(j)*E3+thetadot(j)*e1p+phidot(j)*e3;
    
    T(j) = 0.5*m*(u1(j)^2+u2(j)^2+u3(j)^2)+0.5*dot(I*omega,omega);
    E(j) = T(j)+m*g*x3(j);

end

figure()
hold on
title('Total Energy of Hoop','Interpreter','Latex')
plot(E,'LineWidth',1,'Color','b')

% plot(arr(Atouching_contact_idx2),E(Atouching_contact_idx2),'.', 'LineWidth',1,'Color','r')
% plot(arr(Atouching_contact_idx1),E(Atouching_contact_idx1),'.', 'LineWidth',1,'Color','r')

temp_ones = ones(1,ntime);

plot(arr(Atouching_contact_idx1),0.9*temp_ones(Atouching_contact_idx1),'.', 'LineWidth',1,'Color','r')
plot(arr(Atouching_contact_idx2),-0.9*temp_ones(Atouching_contact_idx2),'.', 'LineWidth',1,'Color','r')

plot(arr(Btouching_contact_idx1),temp_ones(Btouching_contact_idx1),'.', 'LineWidth',1,'Color','k')
plot(arr(Btouching_contact_idx2),-temp_ones(Btouching_contact_idx2),'.', 'LineWidth',1,'Color','k')

plot(arr(Ctouching_contact_idx1),1.1*temp_ones(Ctouching_contact_idx1),'.', 'LineWidth',1,'Color','b')
plot(arr(Ctouching_contact_idx2),-1.1*temp_ones(Ctouching_contact_idx2),'.', 'LineWidth',1,'Color','b')

plot(arr(Dtouching_contact_idx1),1.2*temp_ones(Dtouching_contact_idx1),'.', 'LineWidth',1,'Color','g')
plot(arr(Dtouching_contact_idx2),-1.2*temp_ones(Dtouching_contact_idx2),'.', 'LineWidth',1,'Color','g')

plot(arr(Etouching_contact_idx1),1.3*temp_ones(Etouching_contact_idx1),'.', 'LineWidth',1,'Color','k')
plot(arr(Etouching_contact_idx2),-1.3*temp_ones(Etouching_contact_idx2),'.', 'LineWidth',1,'Color','k')




legend('$E$','A(1) = 1','A(2) = 1','B(1) = 1','B(2) = 1','C(1) = 1','C(2) = 1','D(1) = 1','D(2) = 1','E(1) = 1','E(2) = 1', 'Interpreter', 'latex');

