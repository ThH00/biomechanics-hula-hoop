% Poincare sections of equation (2.14)
close all

% EOM parameters
% m = 11;
% delta = linspace(0,0.01,m);
% alpha = linspace(0,0.4,m);
% beta = linspace(-0.01,0.001,m);


m = 1;
alpha = 0.4;
beta = 0.02;
delta = 0.002;

[Delta, Alpha, Beta] = ndgrid(delta, alpha, beta);
M = m^3;    % total parameter combinations

% choosing different IC
colormap('parula')
nIC = 11;   % number of inital conditions
% y0 = [xi0, dxi_dtau0];
y0 = [linspace(0,2*pi,nIC)', zeros(nIC,1);
    linspace(0,2*pi,nIC)', ones(nIC,1);
    linspace(0,2*pi,nIC)', 2*ones(nIC,1);
    linspace(0,2*pi,nIC)', 3*ones(nIC,1);
    linspace(0,2*pi,nIC)', 4*ones(nIC,1);
    linspace(0,2*pi,nIC)', 5*ones(nIC,1);
    linspace(0,2*pi,nIC)', -ones(nIC,1);
    linspace(0,2*pi,nIC)', -2*ones(nIC,1);
    linspace(0,2*pi,nIC)', -3*ones(nIC,1);
    linspace(0,2*pi,nIC)', -4*ones(nIC,1)];
cmap = parula(nIC);     % color with a uniform map

% choosing different IC
colormap('parula')
cmap = parula(10*nIC);     % color with a uniform map

figure(1);
hold on
% axes title
xlabel("$\xi$",'Interpreter','Latex')
ylabel("$\xi'$",'Interpreter','Latex')
grid on;
axis equal;
box on;

xlim([0,2*pi])
ylim([-6, 6])

opts = odeset('RelTol',1e-9,'AbsTol',1e-9);

% Time setup
omega = 1;
T = 2*pi / omega;     % Forcing period
nPeriods = 100;       % Total number of periods
dt = 0.001;            % Integration time step
tSpan = 0:dt:nPeriods*T;

animation = VideoWriter('poincare_pendulum.mp4', 'MPEG-4');
animation.FrameRate = 10;
open(animation);

% looping over different eom parameters
for r = 1:m
    for s = 1:m
        for t = 1:m

            % looping over different IC
            for k = 1:10*nIC
                [tau, Y] = ode45(@(tau,y) odefcn(tau,y,Alpha(r,s,t),Beta(r,s,t),Delta(r,s,t)), tSpan, y0(k,:), opts);

                % Sample the solution at each period of the forcing (Poincaré section)
                sampleTimes = 0:T:(nPeriods-1)*T;
                xi = mod(interp1(tau, Y(:,1), sampleTimes),2*pi);
                xi_prime = interp1(tau, Y(:,2), sampleTimes);

                % figure title
                title(['$\delta = ', num2str(Delta(r,s,t)), ', \alpha = ', num2str(Alpha(r,s,t)),', \beta = ', num2str(Beta(r,s,t)),'$'], 'Interpreter', 'latex')

                plot(xi, xi_prime,'.','LineWidth',0.5,'Color','r')

                drawnow
                frame = getframe(gcf);  % Capture the current figure frame
                for g = 1:30
                    writeVideo(animation, frame);
                end
                % pause(0.4)
                plot(xi, xi_prime,'.','LineWidth',0.5,'Color',cmap(k,:))

                hold on
                grid on;
                axis equal;
                box on;
                xlim([0,2*pi])
                ylim([-6,6])

                drawnow
                writeVideo(animation, getframe(gcf))

            end
           
            clf

        end
    end
end

close(animation)


% ODE function
function dydt = odefcn(tau,y,alpha,beta,delta)
  dydt = zeros(2,1);
  dydt(1) = y(2);
  dydt(2) = -delta * y(2) - delta + alpha * sin(y(1)) + beta * sin(y(1) + 2 * tau);
end