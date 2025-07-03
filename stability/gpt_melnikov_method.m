% Parameters
alpha = 0.4;
beta = 0.004;
delta = 0.001;

% syms alpha beta delta real

t0_vals = linspace(0, 2*pi, 500);  % range of t0 values

% Homoclinic orbit:
x0 = @(t) 4*atan(tanh(sqrt(alpha)*t))+pi;
dotx0 = @(t) 4*sqrt(alpha)*sech(4*sqrt(alpha)*t);

% Melnikov function
M = zeros(size(t0_vals));
for i = 1:length(t0_vals)
    t0 = t0_vals(i);
    
    integrand_damping = @(t) (dotx0(t)).^2+dotx0(t);
    integrand_forcing = @(t) sin(x0(t)+2*t+2*t0).*dotx0(t);
    
    % Integrate from -T to T as an approximation
    T = 100000;
    damping_term = integral(integrand_damping, -T, T);
    forcing_term = integral(integrand_forcing, -T, T);
    
    M(i) = -delta * damping_term + beta * forcing_term;
end

% Plot Melnikov function
plot(t0_vals, M, 'LineWidth', 2)
xlabel('$t_0$', 'Interpreter', 'latex')
ylabel('$M(t_0)$', 'Interpreter', 'latex')
title('Melnikov Function for Forced Damped Pendulum')
grid on
