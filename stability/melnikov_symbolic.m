clear all
clc

syms xi
syms alpha
f = 1/sqrt(2*alpha*(1-cos(xi)));
area = int(f, xi);

%% plotting the heteroclinic orbit that I got


%% symbolic melnikov analysis
clear all

syms alpha real
syms beta real
syms delta real

syms tau real
syms tau0 real

% Homoclinic orbit:
xi0_plus = @(tau) 4*atan(sqrt(alpha)*(tau-tau0));
xidot0_plus = @(tau) 4*sqrt(alpha)*exp(sqrt(alpha)*(tau-tau0))./(1+exp(2*sqrt(alpha)*(tau-tau0)));

xi0_minus = @(tau) 4*atan(-sqrt(alpha)*(tau-tau0));
xidot0_minus = @(tau) -4*sqrt(alpha)*exp(-sqrt(alpha)*(tau-tau0))./(1+exp(-2*sqrt(alpha)*(tau-tau0)));

integrand_damping_plus = @(tau) (xidot0_plus(tau)).^2+xidot0_plus(tau);
integrand_forcing_plus = @(tau) sin(xi0_plus(tau)+2*tau+2*tau0).*xidot0_plus(tau);

integrand_damping_minus = @(tau) (xidot0_minus(tau)).^2+xidot0_minus(tau);
integrand_forcing_minus = @(tau) sin(xi0_minus(tau)+2*tau+2*tau0).*xidot0_minus(tau);

damping_term_plus = int(integrand_damping_plus, tau, 0, inf);
forcing_term_plus = int(integrand_forcing_plus, tau, 0, inf);

damping_term_minus = int(integrand_damping_minus, tau, -inf, 0);
forcing_term_minus = int(integrand_forcing_minus, tau, -inf, 0);

% Melnikov function
M = -delta * damping_term_minus + beta * forcing_term_minus-delta * damping_term_plus + beta * forcing_term_plus;

M_val = M;
M_val = subs(M_val,alpha,0.4);
M_val = subs(M_val,beta,0.004);
M_val = subs(M_val,delta,0.001);

n = 10;
tau0_vals = linspace(0, 2*pi, n);  % range of t0 values
M_numeric = zeros(n,1);
for i = 1:n
    temp = subs(M_val,tau0,tau0_vals(i));
    M_numeric(i) = double(vpa(temp));
end

figure()
hold on
grid on
box on
plot(M_numeric,'k','Linewidth',2)