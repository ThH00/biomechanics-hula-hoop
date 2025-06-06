clear all
close all
clc

alpha = 0.4;
beta = 0.001;
delta = 0.001;

R = 1.6;
r = 0.4;
omega = 1;
m = 1;

k = delta*2*m*R^2*omega;
a = 2*(R-r)*(alpha+beta);
b = 2*(R-r)*(alpha-beta);

E1 = [1,0];
E2 = [0,1];

tspan = linspace(0,100,10000);
y0 = [pi, 0];
[t,y] = ode45(@(t,y) odefcn(t,y,alpha,beta,delta), tspan, y0);

phi = mod(y(:,1),2*pi);
phiprime = y(:,2);

figure(1)
hold on
title('phase digram')
xlabel('phi')
ylabel('phi-prime')
plot(phi,phiprime,'.','LineWidth',1,'Color','k')

figure(2)
title('trajectories of hip and hoop centers')
axis equal
x1_hip = a*cos(omega*t);
x2_hip = b*sin(omega*t);
er = cos(phi)*E1+sin(phi)*E2;
x_hoop = [x1_hip,  x2_hip]-(R-r)*er;
plot(x1_hip,x2_hip,'LineWidth',1,'Color','k')
hold on
plot(x_hoop(:,1),x_hoop(:,2),'LineWidth',1,'Color','b')
xlabel('x')
ylabel('y')
legend('X-hip','X-hoop')

%%

figure(3)
n = 10;
phi0 = linspace(-2*pi,2*pi,n);

alpha = 0.4;
beta = 0;
delta = 0;


% choosing different IC
colormap('parula')
cmap = parula(n);     % color with a uniform map

tspan = linspace(0,50,1000);

for i = 1:n

    y0 = [phi0(i), 0];
    [t,y] = ode45(@(t,y) odefcn(t,y,alpha,beta,delta), tspan, y0);
    phi = mod(y(:,1),2*pi);
    phiprime = y(:,2);
    
    hold on
    title('phase digram')
    xlabel('phi')
    ylabel('phi-prime')
    plot(phi,phiprime,'.','LineWidth',1,'Color',cmap(i,:))

end


function dydt = odefcn(t,y,alpha,beta,delta)
  dydt = zeros(2,1);
  dydt(1) = y(2);
  dydt(2) = -delta*y(2)-alpha*sin(y(1)-t)+beta*sin(y(1)+t);
end