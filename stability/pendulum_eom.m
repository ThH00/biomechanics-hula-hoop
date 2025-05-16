% Parameters
delta = 1;
alpha = 1;
beta = 0.8;

% System of ODEs
f = @(t, x) [ ...
    x(2);
    -delta * x(2) - delta + alpha * sin(x(1)) + beta * sin(x(1) + 2 * t)
];

% Time span and initial conditions
tspan = [0 100];
x0 = [0; 0];  % Initial conditions: xi(0) = 0, xi'(0) = 0

% Solve ODE
[t, x] = ode45(f, tspan, x0);

% Plot results
figure;
plot(t, x(:,1));
xlabel('Time');
ylabel('\xi(t)');
title('Solution of \xi'''' + \delta\xi'' + \delta - \alpha sin(\xi) = \beta sin(\xi + 2\tau)');
grid on;

figure;
plot(x(:,1), x(:,2));
xlabel('\xi(t)');
ylabel('$\dot{\xi}(t)$','Interpreter','Latex');
title('Solution of \xi'''' + \delta\xi'' + \delta - \alpha sin(\xi) = \beta sin(\xi + 2\tau)');
grid on;