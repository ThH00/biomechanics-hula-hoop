m = 1;
l = 1;
g = 9.81;

epsilon = 0.1;

% n = 10;
% q0 = linspace(-2*pi,2*pi,n);
% m = 5;
% p0 = linspace(-10,10,m);

n = 1;
m = 1;
q0 = 0.9*pi;
p0 = 0;

figure()
hold on
for i = 1:n
    for j = 1:m
    
        tspan = [0 10];
        y0 = [q0(i) p0(j)];
        [t,y] = ode45(@(t,y) simple_pendulum(t,y,m,l,g), tspan, y0);
        [t_e,y_e] = ode45(@(t_e,y_e) forced_simple_pendulum(t_e,y_e,m,l,g,epsilon), tspan, y0);
        [t_d,y_d] = ode45(@(t_d,y_d) forced_damped_simple_pendulum(t_d,y_d,m,l,g,epsilon), tspan, y0);
        
        plot(y(:,1),y(:,2),'k','LineWidth',2)
        hold on
        plot(y_e(:,1),y_e(:,2),'b','LineWidth',2)
        hold on
        plot(y_d(:,1),y_d(:,2),'r','LineWidth',2)

    end
end

function dydt = simple_pendulum(t,y,m,l,g)
  dydt = zeros(2,1);
  dydt(1) = y(2)/(m*l^2);
  dydt(2) = -m*g*l*sin(y(1));
end

function dydt = forced_simple_pendulum(t,y,m,l,g,epsilon)
    % taking omega=1 so theta=t
  dydt = zeros(2,1);
  dydt(1) = y(2)/(m*l^2)+epsilon*sin(t);
  dydt(2) = -m*g*l*sin(y(1))+epsilon*cos(t);
end

function dydt = forced_damped_simple_pendulum(t,y,m,l,g,epsilon)
    % taking omega=1 so theta=t
  dydt = zeros(2,1);
  dydt(1) = y(2)/(m*l^2)+epsilon*sin(t);
  dydt(2) = -m*g*l*sin(y(1))+epsilon*cos(t);
end