clear all; close all; clc;
% Simulation of object motion.

rng(90)  % Seed for random numbers
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

% This figure plots object motion trajectory
figure(1)
plot(z(1,:), z(2,:))
grid minor; axis square
title('Object motion trajectory')
