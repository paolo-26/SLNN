clear all; close all; clc;
% Simulation of object motion.

rng(44)  % Seed for random numbers
N = 100;  % Number of time instants
DELTA = 1.5;
A = [1 0 DELTA 0; 0 1 0 DELTA; 0 0 1 0; 0 0 0 1];

sigmaQx = 2;
sigmaQv = 0.5;

epsilon = zeros(4,N);
epsilon(1:2,:) = sigmaQx * randn(2,N);
epsilon(3:4,:) = sigmaQv * randn(2,N);
z = zeros(4,N);  % State vector (over time)
z(:,1) = [0 0 DELTA DELTA].'; % Initial state: coordinates at time 0 are (0,0)
for i = 2 : N
    z(:,i) = A * z(:,i-1) + epsilon(:,i);
end

C = [1 0 0 0; 0 1 0 0];
sigmaR = 20;
delta = sigmaR * randn(2,N);
y = zeros(2,N);
y(:,1) = [0 0].';
for i = 2 : N
    y(:,i) = C * z(:,i) + delta(:,i);
end



%% Kalman Filter
sigmaQx = 0.2;  % Variance of model
sigmaQv = 0.2;  % Variance of model
sigmaR = 20;  % Variance of observed data

mut = zeros(4, N);
mut(:,1) = [0,0,1.5,1.5]';
%mut(:,1) = [78, 5, 45, 1]';
Qt = diag([sigmaQx, sigmaQx, sigmaQv, sigmaQv]);
Rt = diag([sigmaR; sigmaR]);
sigmat = eye(4);
%sigmat = 100*rand(4,4);
pzt = zeros(4,N);
pzt(:,1) = [0 0 DELTA DELTA].'; 

for i = 2:N
    mut_tp = A*mut(:,i-1);
    sigmat_tp = A * sigmat * A.' + Qt;
    yhat(:,i) = C * mut_tp;
    Kt = sigmat_tp * C' * inv(C*sigmat_tp*C.' + Rt);
    mut(:,i) = mut_tp + Kt * (y(:,i) - yhat(:,i));
    sigmat = (eye(4) - Kt*C) * sigmat_tp;
    %pzt(:,i) = sigmat*randn(mut,1);
end


%% Plots
% This figure plots object motion trajectory.
figure(1)
plot(z(1,:), z(2,:))
grid minor; axis square
xlabel('\it x'); ylabel('\it y')
title('Object motion trajectory')

figure(2)
hold on
grid on
grid minor
xlabel('\it t'); ylabel('\it x');
title('X position vs time')
plot(y(1,:))
plot(yhat(1,:))
plot(mut(1,:),'k:', 'linewidth',1.5)
plot(z(1,:))
legend('Observed','Estimated','Filtered','Real', 'location', 'best')

figure(3)
hold on
grid on
grid minor
xlabel('\it t'); ylabel('\it y');
title('Y position vs time')
plot(y(2,:), 'color', [0, 0.4470, 0.7410])
plot(yhat(2,:), 'color', [0.6350, 0.0780, 0.1840])
plot(mut(2,:),'k:', 'linewidth',1.5)
plot(z(2,:), 'color', [0.9290, 0.6940, 0.1250])
legend('Observed','Estimated','Filtered','Real', 'location', 'best')

% figure(4)
% hold on
% grid on
% grid minor
% xlabel('\it t'); ylabel('\it y');
% title('Velocity_x vs time')
% plot(mut(3,:),':', 'linewidth',2)
% plot(z(3,:), 'linewidth',2)
% legend('Filtered','Real', 'location', 'best')
% 
% figure(5)
% hold on
% grid on
% grid minor
% xlabel('\it t'); ylabel('\it x');
% title('Velocity_y vs time')
% plot(mut(4,:),':', 'linewidth',2)
% plot(z(4,:), 'linewidth',2)
% legend('Filtered','Real', 'location', 'best')

