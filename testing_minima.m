% x = [1 3 7 1 2 6 4 5 3 0 8 7]; % Sample 1D array
x = linspace(-pi/3,8*pi/3,100);
y = cos(x);

% Find local maxima
local_max_idx = islocalmax(y);

% Find local minima
local_min_idx = islocalmin(y);

% Extract maxima and minima values
local_max_values = y(local_max_idx);
local_min_values = y(local_min_idx);

% Display results
disp('Local maxima:');
disp([find(local_max_idx)', local_max_values']);

disp('Local minima:');
disp([find(local_min_idx)', local_min_values']);

figure()
plot(y)
hold on
plot(find(local_max_idx), local_max_values,'*','Color','r');
plot(find(local_min_idx), local_min_values,'*','Color','b');