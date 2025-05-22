E1 = [1,0];
E2 = [0,1];

delta = 0;
alpha = 1;
beta = 1;

R = 1.6;    % hoop radius
r = 0.4;    % hip radius

phi0 = pi/6;
dphi_dtau0 = 1;

tspan = linspace(0,50,1000);
y0 = [phi0; dphi_dtau0];
[tau,y] = ode45(@(tau,phi) eom(tau,phi,delta,alpha,beta),tspan,y0);

phi = y(:,1);
dphi_dtau = y(:,2);
dtheta_dau = (1-r/R)*dphi_dtau;
theta = (1-r/R)*phi;

% coordinates of hip center
a = 2*(alpha+beta)*(R-r);
b = 2*(alpha-beta)*(R-r);
x_hip = a*cos(tau);
y_hip = b*sin(tau);

ang_arr = linspace(0,2*pi,100);


animation = VideoWriter('unsteady_motion_1.mp4', 'MPEG-4');
animation.FrameRate = 10;
open(animation);

figure()
axis([-8,8,-4,4])
box on
axis equal
hold on
for i=1:length(tau)
    % plotting the hip
    hip_x = x_hip(i)+r*cos(ang_arr+phi(i));
    hip_y = y_hip(i)+r*sin(ang_arr+phi(i));
    hip = plot(hip_x,hip_y,'k','LineWidth',1);
    hip_center = plot(x_hip(i),y_hip(i),'.','Color','k');
    hip_marker = plot(hip_x(1),hip_y(1),'k','Marker','*','LineWidth',1);

    % {er, ephi} basis
    er = cos(phi(i))*E1+sin(phi(i))*E2;
    ephi = cos(phi(i))*E2-sin(phi(i))*E1;

    x_hoop = x_hip(i)-(R-r)*er(1);
    y_hoop = y_hip(i)-(R-r)*er(2);

    hoop_center = plot(x_hoop,y_hoop,'.','Color','b');

    % plotting the hoop
    hoop_x = x_hoop+R*cos(ang_arr+theta(i));
    hoop_y = y_hoop+R*sin(ang_arr+theta(i));

    hoop = plot(hoop_x,hoop_y,'b','LineWidth',1);
    hoop_marker = plot(hoop_x(1),hoop_y(1),'b','Marker','*','LineWidth',1);

    axis([-8,8,-4,4])

    drawnow
    writeVideo(animation, getframe(gcf))

    % pause(0.001)

    delete(hip)
    delete(hoop)
    delete(hoop_marker)
    delete(hip_marker)

end
close(animation)


function dphidtau = eom(tau,phi,delta,alpha,beta)
% circular motion
% setting beta = 0
dphidtau = [phi(2);
    -delta*phi(2)+alpha*sin(phi(1)-tau)+beta*sin(phi(1)+tau)];
end