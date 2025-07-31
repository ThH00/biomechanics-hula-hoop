output_directory = "/Users/theresahonein/Desktop/terryhonein/Research-HulaHoop/outputs/2025-07-30_19-50-41";

% Loading the outputs of the generalized-alpha algorithm
load(output_directory+"/q.mat")

% Loading the coordinates of the center of the hip
load(output_directory+"/xbar_hip.mat")
xbar_hip = xbar_hip';

%% Animation
figure()
hold on
view(0,30)
axis equal
box on

%% Plotting cone prep
% Parameters
hib_base_radius = cot(pi/3);  % base radius
hip_height = 1;         % height
hip_n = 50;             % resolution

% Meshgrid in polar coordinates
gamma = linspace(0, 2*pi, hip_n);
z = linspace(0, hip_height, hip_n);
[Gamma, Z] = meshgrid(gamma, z);

% Radius decreases linearly with height
R = hib_base_radius * (1 - Z/hip_height);

% Convert to Cartesian coordinates
X = R .* cos(Gamma);
Y = R .* sin(Gamma);

%% Animating Hoop
figure()
hold on

animation = VideoWriter(output_directory+"/animation.mp4", 'MPEG-4');
animation.FrameRate = 30;
open(animation);

% Fixed basis
E1 = [1;0;0];
E2 = [0;1;0];
E3 = [0;0;1];

ang_arr = linspace(0,2*pi,100);
R_hoop = 1.2;

k = 1;

view_array = [15,15;90,0;0,90];
% isometric, front, top

for p = 1:3
    subplot(1,3,p)

    % Plot cone
    surf(X, Y, Z)
    shading interp
    colormap turbo
end

for i = 1:600 %1:length(q(k,1,:))
    
    % view(0,30)
    % view(1)
    % xlim([-2, 2])
    % ylim([-2, 2])
    % zlim([-0.5, 2])

    x1 = q(k,1,i);
    x2 = q(k,2,i);
    x3 = q(k,3,i);

    psi = q(k,4,i);
    theta = q(k,5,i);
    phi = q(k,6,i);
    
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

    title(num2str(i));

    for p = 1:3

        subplot(1,3,p)  
        hold on

        % % Plot cone
        % surf(X, Y, Z)
        % shading interp
        % colormap turbo

        % Plot hoop
        center_plot(p) = plot3(x1, x2, x3, 'Marker', '.', 'MarkerSize', 10);
    
        e1_plot(p) = quiver3(x1, x2, x3, e1(1), e1(2), e1(3),'r');
        e2_plot(p) = quiver3(x1, x2, x3, e2(1), e2(2), e2(3),'g');
        e3_plot(p) = quiver3(x1, x2, x3, e3(1), e3(2), e3(3),'b');
        
        % plotting the hoop
        circle(p) = plot3(x1+R_hoop*cos(ang_arr)*e1(1)+R_hoop*sin(ang_arr)*e2(1), ...
            x2+R_hoop*cos(ang_arr)*e1(2)+R_hoop*sin(ang_arr)*e2(2), ...
            x3+R_hoop*cos(ang_arr)*e1(3)+R_hoop*sin(ang_arr)*e2(3), ...
            'color','b','LineWidth',2);
    
        axis equal
        xlim([-2, 2])
        ylim([-2, 2])
        zlim([-0.5, 2])
        
        xlabel('X')
        ylabel('Y')
        zlabel('Z')
        view(view_array(p,1),view_array(p,2))

    end

    % draw the minimzing points on hoop and hip
    % for j = 1:2
    %     u = cos(tau(j,i))*e1+sin(tau(j,i))*e2;
    %     xM = [x1; x2; x3]+R_hoop*u;
    %     dv = xM(3)
    %     temp = xM-dv*E3-xbar_hip(:,i);
    %     v = (temp)/norm(temp);
    %     xP = xbar_hip(:,i)+dv*E3+R_hip*v;
    %     min_hoop(j) = plot3(xM(1),xM(2),xM(3),'*','Color','r','LineWidth',2);
    %     min_hip(j) = plot3(xP(1),xP(2),xP(3),'*','Color','r','LineWidth',2);
    % end

    drawnow
    writeVideo(animation, getframe(gcf))

    % pause(0.001)
    for p = 1:3
        subplot(1,3,1)
        delete(center_plot(p))
        delete(e1_plot(p))
        delete(e2_plot(p))
        delete(e3_plot(p))
        delete(circle(p))
    end

    % delete(min_hoop)
    % delete(min_hip)

end

close(animation)
% 
% load('gN.mat')
% figure()
% plot(gN(1,:))
% hold on
% plot(gN(2,:))

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

%% 
z = q(k,3,:);
z = squeeze(z); 
plot(z,'.')


