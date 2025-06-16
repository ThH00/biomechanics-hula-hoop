syms alpha tau t0 real

integrand = 2 * sqrt(alpha) * sech(sqrt(alpha) * tau) * sin(2 * acos(tanh(-sqrt(alpha) * tau)) + 2 * tau + 2 * t0);

result = int(integrand, tau, -inf, inf);

disp(result);