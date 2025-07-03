% hoop parameters
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

% Time setup
T = 2*pi / omega;     % Forcing period
nPeriods = 500;       % Total number of periods
dt = 0.001;            % Integration time step
tSpan = 0:dt:nPeriods*T;

% Initial conditions
y0 = [pi; 0];         % [x0; v0]

% Integrate using ode45
opts = odeset('RelTol',1e-9,'AbsTol',1e-9);
[t, Y] = ode45(@(t,y) odefcn(t,y,alpha,beta,delta), tSpan, y0, opts);

% Sample the solution at each period of the forcing (Poincaré section)
sampleTimes = 0:T:(nPeriods-1)*T;
phi = mod(interp1(t, Y(:,1), sampleTimes),2*pi);
phi_prime = interp1(t, Y(:,2), sampleTimes);

% Plot the Poincaré section
figure(1);
hold on
plot(phi, phi_prime, 'b.', 'MarkerSize', 4);
xlabel('x');
ylabel('v = dx/dt');
title('Poincaré Section of Driven Hula Hoop');
grid on;
axis equal;

% ODE function
function dydt = odefcn(t,y,alpha,beta,delta)
  dydt = zeros(2,1);
  dydt(1) = y(2);
  dydt(2) = -delta*y(2)-alpha*sin(y(1)-t)+beta*sin(y(1)+t);
end