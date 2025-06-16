%% plotting the heteroclinic orbits
tau = -10:0.1:10;

alpha = 1;

xi_plus = 2*acos(tanh(-sqrt(alpha)*tau));
xi_prime_plus = 2*sqrt(alpha)*sech(sqrt(alpha)*tau);

xi_minus = 2*acos(tanh(sqrt(alpha)*tau));
xi_prime_minus = -2*sqrt(alpha)*sech(sqrt(alpha)*tau);

figure()
hold on
box on
grid on
xlabel('$\xi$','Interpreter','Latex')
ylabel('$\frac{d\xi}{d\tau}$','Interpreter','Latex')
plot(xi_plus,xi_prime_plus,'k','LineWidth',1);
plot(xi_minus,xi_prime_minus,'b','LineWidth',1);

%% plotting g(alpha) - OOR calculations
alpha = 0:0.1:100;
g = 4./(alpha.*(4*sqrt(alpha)/pi+1)).*(csch(pi./sqrt(alpha))+sech(pi./sqrt(alpha)));

figure()
hold on
box on
grid on
title("A plot of $g(\alpha)$ using OOR's calculations",'Interpreter','Latex')
xlabel('$\alpha$','Interpreter','Latex')
ylabel('$g(\alpha)$','Interpreter','Latex')
plot(alpha, g)

%% plotting g(alpha) - my calculations
alpha = 0:0.1:1000;
g = 4./(sqrt(alpha).*(4*sqrt(alpha)/pi+1)).*(csch(pi./sqrt(alpha))-sech(pi./sqrt(alpha)));

figure()
hold on
box on
grid on
title("A plot of $g(\alpha)$ using TH calculations",'Interpreter','Latex')
xlabel('$\alpha$','Interpreter','Latex')
ylabel('$g(\alpha)$','Interpreter','Latex')
plot(alpha, g)

%% Numerical integration of A in Matlab
% Parameters
alpha = 0:100;
n = size(alpha,2);
t0 = pi/4;

result = zeros(1,n);

for i = 1:n

    % Integrand as an anonymous function
    integrand = @(tau) 2*sqrt(alpha(i)) .* sech(sqrt(alpha(i))*tau) .* ...
        sin(2*acos(tanh(-sqrt(alpha(i))*tau)) + 2*tau + 2*t0);
    
    % Perform numerical integration
    result(i) = integral(integrand, -Inf, Inf, 'RelTol',1e-10,'AbsTol',1e-12);

end

figure()
hold on
box on
grid on
title("A plot of the integral A using Matlab numerical calculations",'Interpreter','Latex')
xlabel('$\alpha$','Interpreter','Latex')
ylabel('$A$','Interpreter','Latex')
plot(alpha, result,'k')


A_symbolic = 8*pi./alpha.*(csch(pi./sqrt(alpha))-sech(pi./sqrt(alpha)))*sin(2*t0);
hold on
plot(alpha, A_symbolic,'b')
