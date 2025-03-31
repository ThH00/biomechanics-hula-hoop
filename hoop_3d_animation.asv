% loading the outputs of the generalized-alpha algorithm
load("q.mat")

% loading the coordinates of the center of the hip
load("xbar_hip.mat")
xbar_hip = xbar_hip';

% loading the minizing values of tau and dv
load("tau.mat")

figure()
hold on
view(0,30)
% view(1)
% axis equal
xlim([-2, 2])
ylim([-2, 2])
zlim([-2, 2])
box on

% Fixed basis
E1 = [1;0;0];
E2 = [0;1;0];
E3 = [0;0;1];

animation = VideoWriter('3D_hoop3_2contacts_moving_hip_3_3D.mp4', 'MPEG-4');
animation.FrameRate = 10;
open(animation);

ang_arr = linspace(0,2*pi,100);
R_hoop = 0.5;
R_hip = 0.2;

% plot the cone
z = linspace(-1.5C,1.5,100);
% r = 0.5-0.4*z;
r = R_hip*ones(100,1);
hold on

for i = 1:205 %length(q)

    
    % view(0,30)
    % view(1)
    xlim([-2, 2])
    ylim([-2, 2])
    zlim([-2, 2])

    x1 = q(i,1);
    x2 = q(i,2);
    x3 = q(i,3);

    psi = q(i,4);
    theta = q(i,5);
    phi = q(i,6);
    
    %% basis vectors
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

    center_plot = plot3(x1, x2, x3, 'Marker', '.', 'MarkerSize', 10);

    e1_plot = quiver3(x1, x2, x3, e1(1), e1(2), e1(3),'r');
    e2_plot = quiver3(x1, x2, x3, e2(1), e2(2), e2(3),'g');
    e3_plot = quiver3(x1, x2, x3, e3(1), e3(2), e3(3),'b');

    % plotting the hip
    for j = 1:2:length(ang_arr)
        hip(j) = plot3(xbar_hip(1,i)+r*cos(ang_arr(j)),xbar_hip(2,i)+r*sin(ang_arr(j)),z,'k');
    end

    [circle_hip, center_hip, angle_hip] = plot_circle(0.2, [xbar_hip(1,i), xbar_hip(2,i)], 0, 'k');

    % plotting the hoop
    circle = plot3(x1+R_hoop*cos(ang_arr)*e1(1)+R_hoop*sin(ang_arr)*e2(1), ...
        x2+R_hoop*cos(ang_arr)*e1(2)+R_hoop*sin(ang_arr)*e2(2), ...
        x3+R_hoop*cos(ang_arr)*e1(3)+R_hoop*sin(ang_arr)*e2(3), ...
        'color','b','LineWidth',2);

    % draw the minimzing points on hoop and hip
    for j = 1:2
        u = cos(tau(j,i))*e1+sin(tau(j,i))*e2;
        xM = [x1; x2; x3]+R_hoop*u;
        dv = xM(3)
        temp = xM-dv*E3-xbar_hip(:,i);
        v = (temp)/norm(temp);
        xP = xbar_hip(:,i)+dv*E3+R_hip*v;
        min_hoop(j) = plot3(xM(1),xM(2),xM(3),'*','Color','r','LineWidth',2);
        min_hip(j) = plot3(xP(1),xP(2),xP(3),'*','Color','r','LineWidth',2);
    end

    drawnow
    writeVideo(animation, getframe(gcf))

    pause(0.001)

    delete(center_plot)
    delete(e1_plot)
    delete(e2_plot)
    delete(e3_plot)
    delete(circle)
    delete(hip)

    delete(circle_hip)
    delete(angle_hip)

    delete(min_hoop)
    delete(min_hip)

end

close(animation)

load('gN.mat')
figure()
plot(gN(1,:))
hold on
plot(gN(2,:))

figure()
subplot(1,3,1)
title('psi')
plot(q(:,4))

subplot(1,3,2)
title('theta')
plot(q(:,5))

subplot(1,3,3)
title('phi')
plot(q(:,6))



