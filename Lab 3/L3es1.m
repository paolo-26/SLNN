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

N = SPLIT1 + SPLIT2
c = 0;
x1 = [class1train' class2train']
for k = 1:SPLIT1+SPLIT2
    c = c + x1(:,k) * x1(:,k)';
end

covariance1 = c/N;
covariance2 = 1/N * x1(:,:) * x1(:,:)';
covariance3 = cov([class1train; class2train]);

[V,D] = eig(covariance1);

o = 1
test = [class1test ; class2test]'

for K=1:220
    W = V(:,end-K+1:end);
    z = W'*test;
    xHat = W*z;
    MSE(o) = norm(xHat - test)^2/length(test);
    o = o + 1;
end

figure(1)
hold on
plot(MSE)
grid on
title('Mean squared error, the real one')
legend('MSE (test)')
grid minor
xlim([1 220])
xlabel('K')
ylabel('MSE')

val = diag(D)
val(:,2)=1:length(D)
val = sortrows(val,1, 'descend')
index = [val(1:3,2)]

figure(2)
plot(V(:,index))
grid on
grid minor
title('Most important eigenvectors')
xlim([1 220])