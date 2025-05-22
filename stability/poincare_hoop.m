% from chatgpt

function poincare_hoop()
    % Parameters
    delta = 1;
    alpha = 1;
    beta = 0.8;

    % Time span
    T = pi;  % forcing period
    N_cycles = 1000;  % number of cycles to simulate
    tspan = [0 N_cycles*T];

    % Initial condition
    phi0 = pi/6;
    dphi_dtau0 = 1;
    y0 = [phi0; dphi_dtau0];

    % Solve ODE
    options = odeset('RelTol',1e-8,'AbsTol',1e-10);
    [~, ~, poincare_points] = compute_poincare(y0, tspan, T, ...
        @(t, y) hoop(t, y, delta, alpha, beta), options);

    % Plot the Poincaré map
    figure;
    plot(poincare_points(:,1), poincare_points(:,2), 'k.', 'MarkerSize', 5);
    xlabel('x');
    ylabel('y');
    title('Poincaré Map of Hula Hoop');
    grid on;
end

function dydtau = hoop(tau,y,delta,alpha,beta)
% circular motion
% setting beta = 0
dydtau = [y(2);
    -delta*y(2)+alpha*sin(y(1)-tau)+beta*sin(y(1)+tau)];
end

function [t_all, x_all, poincare_points] = compute_poincare(x0, tspan, T, ode_func, options)
    % Integrate and sample solution stroboscopically every T seconds
    t_current = 0;
    x_current = x0;
    t_all = [];
    x_all = [];
    poincare_points = [];

    while t_current < tspan(2)
        t_next = t_current + T;
        [t_sol, x_sol] = ode45(ode_func, [t_current t_next], x_current, options);
        
        % Interpolate to get state at exactly t_next
        x_interp = interp1(t_sol, x_sol, t_next);
        
        % Store the result
        t_all = [t_all; t_next];
        x_all = [x_all; x_interp];
        poincare_points = [poincare_points; x_interp];  % x_interp is [x, y]
        
        % Update initial conditions
        t_current = t_next;
        x_current = x_interp';
    end
end
