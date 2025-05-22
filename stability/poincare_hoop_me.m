close all

m = 3;

delta = linspace(0,2,m);
alpha = linspace(-2,2,m);
beta = linspace(-2,2,m);

[Delta, Alpha, Beta] = ndgrid(delta, alpha, beta);
M = m^3;

% colors = ['k','b','g','r','y','c','m'];
colors = ['k'];
n_colors = length(colors);

phi0 = pi/6;
dphi_dtau0 = 1;
y0 = [phi0, dphi_dtau0];

N = 10000;    % number of cycles



counter = 1;
for r = 1:m
    for s = 1:m
        for t = 1:m

            figure(r)
            hold on

            subplot(m,m,mod(counter,m^2)+1)

            poincare_points = zeros(N,2);
            poincare_points(1,:) = y0;
            
            for i = 2:N
                tspan = pi*[i-1, i];
                [tau,y] = ode45(@(tau,phi) eom(tau,phi,Delta(r,s,t),Alpha(r,s,t),Beta(r,s,t)),tspan,poincare_points(i-1,:));
                poincare_points(i,:) = y(end,:);
            end
            
            phi = mod(poincare_points(:,1),2*pi);
            phidot = poincare_points(:,2);
            
            plot(phi,phidot,'.','LineWidth',2,'Color',colors(mod(counter,n_colors)+1))
        
            title(['$\delta = ', num2str(Delta(r,s,t)), ', \alpha = ', num2str(Alpha(r,s,t)),', \beta = ', num2str(Beta(r,s,t)),'$'], 'Interpreter', 'latex')
        
            ylim([-4, 4])
            counter = counter+1;

        end
    end
end




function dphidtau = eom(tau,phi,delta,alpha,beta)
% circular motion
% setting beta = 0
dphidtau = [phi(2);
    -delta*phi(2)+alpha*sin(phi(1)-tau)+beta*sin(phi(1)+tau)];
end