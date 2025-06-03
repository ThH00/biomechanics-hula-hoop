% Parameters for the hyperboloid
a = 1;  % x-axis scaling
b = 1;  % y-axis scaling
c = 1;  % z-axis scaling

% Parameter ranges
theta = linspace(0, 2*pi, 100);     % Angle around the z-axis
z = linspace(-2, 2, 100);           % Height along the z-axis

% Create meshgrid
[Theta, Z] = meshgrid(theta, z);

% Parametric equations for a hyperboloid of one sheet
X = a * cosh(Z) .* cos(Theta);
Y = b * cosh(Z) .* sin(Theta);
Z = c * Z;

% Plot the surface
surf(X, Y, Z)
axis equal
xlabel('X'), ylabel('Y'), zlabel('Z')
title('One-Sheet Hyperboloid')
shading interp
colormap jet
