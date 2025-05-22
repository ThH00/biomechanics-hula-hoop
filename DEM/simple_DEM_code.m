% Simple 2D DEM Simulation in MATLAB

clear; clc;

% Parameters
N = 10;                 % Number of particles
radius = 0.05;          % Radius of particles (m)
mass = 0.1;             % Mass of particles (kg)
k_n = 1e4;              % Normal stiffness (N/m)
gamma_n = 5;            % Damping coefficient
dt = 1e-4;              % Time step (s)
steps = 5000;           % Number of time steps
g = 9.81;               % Gravity (m/s^2)
domain = [0 1 0 1];     % [xmin xmax ymin ymax]

% Initialization
pos = rand(N,2) * 0.8 + 0.1;         % Random initial positions
vel = zeros(N,2);                    % Initial velocities
force = zeros(N,2);                  % Force array

% Visualization setup
figure;
hold on; axis equal;
xlim(domain(1:2)); ylim(domain(3:4));

for step = 1:steps
    force(:,:) = 0;  % Reset force

    % Gravity
    force(:,2) = force(:,2) - mass * g;

    % Particle-particle interactions
    for i = 1:N-1
        for j = i+1:N
            rij = pos(j,:) - pos(i,:);
            dist = norm(rij);
            overlap = 2*radius - dist;
            if overlap > 0
                nij = rij / dist;
                vij = vel(j,:) - vel(i,:);
                fn = k_n * overlap * nij - gamma_n * dot(vij, nij) * nij;
                force(i,:) = force(i,:) - fn;
                force(j,:) = force(j,:) + fn;
            end
        end
    end

    % Particle-wall collisions
    for i = 1:N
        % Left wall
        if pos(i,1) - radius < domain(1)
            overlap = domain(1) - (pos(i,1) - radius);
            fn = k_n * overlap - gamma_n * vel(i,1);
            force(i,1) = force(i,1) + fn;
        end
        % Right wall
        if pos(i,1) + radius > domain(2)
            overlap = (pos(i,1) + radius) - domain(2);
            fn = -k_n * overlap - gamma_n * vel(i,1);
            force(i,1) = force(i,1) + fn;
        end
        % Bottom wall
        if pos(i,2) - radius < domain(3)
            overlap = domain(3) - (pos(i,2) - radius);
            fn = k_n * overlap - gamma_n * vel(i,2);
            force(i,2) = force(i,2) + fn;
        end
        % Top wall
        if pos(i,2) + radius > domain(4)
            overlap = (pos(i,2) + radius) - domain(4);
            fn = -k_n * overlap - gamma_n * vel(i,2);
            force(i,2) = force(i,2) + fn;
        end
    end

    % Time integration (Explicit Euler)
    acc = force / mass;
    vel = vel + acc * dt;
    pos = pos + vel * dt;

    % Visualization every 50 steps
    if mod(step,50)==0
        clf;
        hold on;
        for i = 1:N
            viscircles(pos(i,:), radius,'Color','b','LineWidth',0.5);
        end
        xlim(domain(1:2)); ylim(domain(3:4));
        title(['Step: ', num2str(step)]);
        drawnow;
    end
end
