clear; clc; close all;

data = load('Indian_Pines_Dataset')
indian_pines = data.indian_pines;
indian_pines_gt = data.indian_pines_gt;
C1 = 237;  % Corn
C2 = 1265;  % Woods
N_SPECTR = 220;

K = 154

SPLIT = 0.7
SPLIT1 = round(C1*SPLIT)
SPLIT2 = round(C2*SPLIT)


%% TASK 1
n=0;
class1 = zeros(C1, N_SPECTR);
for i = 1:size(indian_pines, 1)
    for j = 1:size(indian_pines, 2)
        if indian_pines_gt(i,j)== 4 % class index
            n = n + 1;
            class1(n,:) = indian_pines(i,j,:);
        end
    end
end

n = 0;
class2 = zeros(C2, N_SPECTR);
for i = 1:size(indian_pines, 1)
    for j = 1:size(indian_pines, 2)
        if indian_pines_gt(i,j)== 14 % class index
            n = n + 1;
            class2(n,:) = indian_pines(i,j,:);
        end
    end
end

class1train = class1(1:SPLIT1,:);
class1test = class1(SPLIT1+1:end,:);

class2train = class2(1:SPLIT2,:);
class2test = class2(SPLIT2+1:end,:);

%% TASK 2 covariance matrix
class1test = class1test - mean(class1train);  % Centering the data (column means)
class2test = class2test - mean(class2train);  % Centering the data (column means)

N = C1 + C2
c1 = 0;
for k = 1:220
    c1 = c1 + class1(k,:)'*class1(k,:);
end
%c1 = c1/C1;

c2 = 0;
for k = 1:220
    c2 = c2 + class2(k,:)'*class2(k,:);
end
%c2 = c1/C2;
covariance1 = (c1 + c2)/N;


%N1 = length(class1(:,1));
sigmaHat1 = class1(:,:)' * class1(:,:);

%N2 = length(class2(:,1));
sigmaHat2 = class2(:,:)' * class2(:,:);

covariance1 = 1/N * (sigmaHat1+sigmaHat2)
covariance2 = sigmaHat1 + sigmaHat2;

%covariance3 = cov(class1) + cov(class2);

[V,D] = eig(covariance1);

o = 1
for K=10:200
    What = V(:,end-K+1:end);
    z = What'*class1test';
    xHat = What*z;
    xHat = xHat';
    MSE(o) = norm(xHat - class1test)^2/71;
    o = o + 1;
end

plot(MSE)
grid minor
legend('MSE, the real one')