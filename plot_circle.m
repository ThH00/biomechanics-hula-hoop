function [circle, center, angle] = plot_circle(r, c, a, color)
%plot_circle plots a circle of radius r and center c and rotation angle a
%       plot_circle(0.5, [0,0], 'k')

% Generate points for the circle
theta = linspace(0, 2*pi, 100); % 100 points from 0 to 2*pi
x = c(1) + r * cos(theta); % X-coordinates
y = c(2) + r * sin(theta); % Y-coordinates

% Plot the circle
circle = plot(x, y, color, 'LineWidth', 2); % Blue circle
angle = plot(c(1) + r * cos(a), c(2) + r * sin(a), '+', 'color', color, 'MarkerSize', 10, 'LineWidth', 2);
hold on;
center = plot(c(1), c(2), '.', 'MarkerSize', 10, 'LineWidth', 2, 'color', color); % Mark the center
% axis equal; % Ensure equal scaling for both axes
grid on;
box on;

end