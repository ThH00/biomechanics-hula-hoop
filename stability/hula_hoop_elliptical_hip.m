clear all
close all
clc

alpha = 0.4;
beta = 0.004;
delta = 0.001;

% g_alpha = pi*(cosec);

R = 1.6;
r = 0.4;
omega = 1;
m = 1;

k = delta*2*m*R^2*omega;
a = 2*(R-r)*(alpha+beta);
b = 2*(R-r)*(alpha-beta);

E1 = [1,0];
E2 = [0,1];

tspan = linspace(0,1000,10000);
y0 = [0.1, 0.1];
[t,y] = ode45(@(t,y) odefcn(t,y,alpha,beta,delta), tspan, y0);

xi = mod(y(:,1),2*pi);
xiprime = y(:,2);

figure()
plot(xi)
xlabel('iter')
ylabel('xi')

figure()
plot(xi,xiprime)
xlabel('xi')
ylabel('xiprime')


figure(1)
hold on
title('phase digram')
xlabel('phi')
ylabel('phi-prime')
plot(xi,xiprime,'.','LineWidth',1,'Color','k')

figure(2)
title('trajectories of hip and hoop centers')
axis equal
x1_hip = a*cos(omega*t);
x2_hip = b*sin(omega*t);
er = cos(xi)*E1+sin(xi)*E2;
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
phi0 = linspace(0,2*pi,n);

alpha = 0.4;
beta = 0.004;
delta = 0.001;
% beta = 0;
% delta = 0;


% choosing different IC
colormap('parula')
cmap = parula(n);     % color with a uniform map

tspan = linspace(0,50,1000);
xlim([0,2*pi])

for i = 1:n

    y0 = [phi0(i), 0];
    [t,y] = ode45(@(t,y) odefcn(t,y,alpha,beta,delta), tspan, y0);
    xi = mod(y(:,1),2*pi);
    xiprime = y(:,2);
    
    hold on
    title('phase digram')
    xlabel('phi')
    ylabel('phi-prime')
    plot(xi,xiprime,'.','LineWidth',1,'Color',cmap(i,:))
    xlim([0,2*pi])

end


function dydt = odefcn(t,y,alpha,beta,delta)
  dydt = zeros(2,1);
  dydt(1) = y(2);
  dydt(2) = -delta-delta*y(2)+alpha*sin(y(1))+beta*sin(y(1)+2*t);
end