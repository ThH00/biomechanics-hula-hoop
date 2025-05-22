% Poincare sections of equation 5.2
close all

% EOM parameters
m = 11;
delta = linspace(0,0.2,m);
alpha = linspace(0,4,m);
beta = linspace(-0.5,0.5,m);

[Delta, Alpha, Beta] = ndgrid(delta, alpha, beta);
M = m^3;    % total parameter combinations

% choosing different IC
colormap('parula')
nIC = 11;   % number of inital conditions
% y0 = [xi0, dxi_dtau0];
y0 = [3*pi/4*ones(nIC,1), linspace(-4,4,nIC)'];
cmap = parula(nIC);     % color with a uniform map

% number of cycles to solve eom for given specific eom parameters and IC
N = 5000;

figure(1)
hold on

animation = VideoWriter('poincare_pendulum_movie_many_IC.mp4', 'MPEG-4');
animation.FrameRate = 10;
open(animation);

% looping over different eom parameters
for r = 1:m
    for s = 1:m
        for t = 1:m

            % looping over different IC
            for k = 1:nIC

                % figure title
                title(['$\delta = ', num2str(Delta(r,s,t)), ', \alpha = ', num2str(Alpha(r,s,t)),', \beta = ', num2str(Beta(r,s,t)),'$'], 'Interpreter', 'latex')
                % axes title
                xlabel('$\xi$','Interpreter','Latex')
                ylabel('$\dot{\xi}$','Interpreter','Latex')
    
                % arrays to stor poincate points for spefici eom paramters
                % and IC
                poincare_points = zeros(N,2);
                poincare_points(1,:) = y0(k,:);
                for i = 2:N
                    % result of one iteration is IC for next
                    tspan = pi*[i-1, i];
                    [tau,y] = ode45(@(tau,phi) eom(tau,phi,Delta(r,s,t),Alpha(r,s,t),Beta(r,s,t)),tspan,poincare_points(i-1,:));
                    poincare_points(i,:) = y(end,:);
                end
                
                % plotting results
                phi = mod(poincare_points(:,1),2*pi);
                phidot = poincare_points(:,2);
                plot(phi,phidot,'.','LineWidth',2,'Color',cmap(k,:))
            
                hold on
                xlim([0,2*pi])
                ylim([-5, 5])

            end

            
            drawnow
            writeVideo(animation, getframe(gcf))
            
            clf

        end
    end
end

close(animation)



function dydtau = eom(tau,xi,delta,alpha,beta)
% forced, damped pendulum
dydtau = [xi(2);
    -delta*xi(2)-delta+alpha*sin(xi(1)-tau)+beta*sin(xi(1)+tau)];
end