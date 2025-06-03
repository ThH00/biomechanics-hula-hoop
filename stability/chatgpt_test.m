% Duffing oscillator parameters
delta = 0.3;      % damping
alpha = -1.0;     % linear stiffness
beta  = 1.0;      % nonlinear stiffness
gamma = 0.6;      % forcing amplitude
omega = 1.5;      % forcing frequency

% Time setup
T = 2*pi / omega;     % Forcing period
nPeriods = 500;       % Total number of periods
dt = 0.001;            % Integration time step
tSpan = 0:dt:nPeriods*T;

% Initial conditions
y0 = [0.1; 0];         % [x0; v0]

% ODE function for Duffing
duffing = @(t, y) [ ...
    y(2);
   -delta*y(2) - alpha*y(1) - beta*y(1)^3 + gamma*cos(omega*t)
];

% Integrate using ode45
opts = odeset('RelTol',1e-9,'AbsTol',1e-9);
[t, Y] = ode45(duffing, tSpan, y0, opts);

% Sample the solution at each period of the forcing (Poincaré section)
sampleTimes = 0:T:(nPeriods-1)*T;
x_sample = interp1(t, Y(:,1), sampleTimes);
v_sample = interp1(t, Y(:,2), sampleTimes);

% Plot the Poincaré section
figure(1);
hold on
plot(x_sample, v_sample, 'b.', 'MarkerSize', 4);
xlabel('x');
ylabel('v = dx/dt');
title('Poincaré Section of Driven Duffing Oscillator');
grid on;
axis equal;
