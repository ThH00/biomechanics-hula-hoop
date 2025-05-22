E1 = [1,0];
E2 = [0,1];

delta = 0.2;
alpha = 0.4;

R = 1.6;    % hoop radius
r = 0.4;    % hip radius

% steady motion 1
tau = linspace(0,8*pi,100);
phi0 = pi-pi/6;
phi = -tau+phi0;
dphi_dtau = -1;
dtheta_dau = (1-r/R)*dphi_dtau;

% coordinates of hip center
a = alpha*4*(R-r)/2;
x_hip = a*cos(tau);
y_hip = a*sin(tau);

ang_arr = linspace(0,2*pi,100);


animation = VideoWriter('steady_motion_4.mp4', 'MPEG-4');
animation.FrameRate = 10;
open(animation);

figure()
axis([-5,5,-5,5])
box on
axis equal
hold on
for i=1:length(tau)
    % plotting the hip
    hip_x = x_hip(i)+r*cos(ang_arr);
    hip_y = y_hip(i)+r*sin(ang_arr);
    hip = plot(hip_x,hip_y,'k','LineWidth',1);
    hip_center = plot(x_hip(i),y_hip(i),'.','Color','k');

    % {er, ephi} basis
    er = cos(phi(i))*E1+sin(phi(i))*E2;
    ephi = cos(phi(i))*E2-sin(phi(i))*E1;

    x_hoop = x_hip(i)-(R-r)*er(1);
    y_hoop = y_hip(i)-(R-r)*er(2);

    hoop_center = plot(x_hoop,y_hoop,'.','Color','b');

    % plotting the hoop
    hoop_x = x_hoop+R*cos(ang_arr);
    hoop_y = y_hoop+R*sin(ang_arr);

    hoop = plot(hoop_x,hoop_y,'b','LineWidth',1);
    hoop_marker = plot(hoop_x(1),hoop_y(1),'b','Marker','*','LineWidth',1);
    
    axis([-5,5,-5,5])

    drawnow
    writeVideo(animation, getframe(gcf))

    pause(0.001)
    
    delete(hip)
    delete(hoop)
    delete(hoop_marker)

end
close(animation)


% tspan = [0 10];
% y0 = [pi/6; 1];
% [tau,phi] = ode45(@(tau,phi) eom(tau,phi,delta,alpha),tspan,y0);
% 
% function dphidtau = eom(tau,phi,delta,alpha)
% % circular motion
% % setting beta = 0
% dphidtau = [phi(2);
%     -delta*phi(2)+alpha*sin(phi(1)-tau)];
% end