clear; clc; close all;

data = load('Indian_Pines_Dataset')
indian_pines = data.indian_pines;
indian_pines_gt = data.indian_pines_gt;
C1 = 1428;  
C2 = 972;
N_SPECTR = 220;

SPLIT = 0.75
SPLIT1 = round(C1*SPLIT)
SPLIT2 = round(C2*SPLIT)


%% Extract data
n=0;
class1 = zeros(C1, N_SPECTR);
for i = 1:size(indian_pines, 1)
    for j = 1:size(indian_pines, 2)
        if indian_pines_gt(i,j)== 2 % class index
            n = n + 1;
            class1(n,:) = indian_pines(i,j,:);
        end
    end
end

n = 0;
class2 = zeros(C2, N_SPECTR);
for i = 1:size(indian_pines, 1)
    for j = 1:size(indian_pines, 2)
        if indian_pines_gt(i,j)== 10 % class index
            n = n + 1;
            class2(n,:) = indian_pines(i,j,:);
        end
    end
end

class1train = class1(1:SPLIT1,:);
class1test = class1(SPLIT1+1:end,:);

class2train = class2(1:SPLIT2,:);
class2test = class2(SPLIT2+1:end,:);
clear class1 class2
%% Covariance matrix
classTrain = ([class1train; class2train]);
mTrain = mean(classTrain);
m1 = mean(class1train);
m2 = mean(class2train);
%class1test = class1test - m1;  % Centering the data (column means)
%class2test = class2test - m2;  % Centering the data (column means)
classTest = [class1test; class2test]';
figure(9)
hold on
%plot(class1test)
%plot(class2test)
N = SPLIT1 + SPLIT2;
c = 0;
x = [class1train' class2train'];
for k = 1:SPLIT1+SPLIT2
    c = c + x(:,k) * x(:,k)';
end

covariance1 = c/N;
covariance2 = 1/N * x(:,:) * x(:,:)';
covariance3 = cov([class1train; class2train]);

%% PCA
[V,D] = eig(covariance1);  % Eigendecomposition

o = 1;
%test = [class1test ; class2test]';
test = classTest - mTrain';


%xHat = sqrt(inv(Lambda))*W'*xHat;
for K=1:220
    Lambda = D(end-K+1:end, end-K+1:end);
    W = V(:,end-K+1:end);
    z = W'*test;
    %z = sqrt(inv(Lambda))*W'*test;  % Whitening
    xHat = W*z + mTrain';
    MSE(o) = (norm(xHat - (test + mTrain')))^2/length(test);
    o = o + 1;
end

figure(1)
hold on
semilogy(MSE)
grid on
title('Mean squared error, the real one')
legend('MSE (test)')
grid minor
xlim([1 220])
xlabel('K')
ylabel('MSE')

val = diag(D);
val(:,2)=1:length(D);
val = sortrows(val,1, 'descend');
index = [val(1:3,2)];

figure(2)
plot(V(:,index))
grid on
grid minor
title('Most important eigenvectors')
xlim([1 220])


%% Optional part
x0 = 0.5 * (m1+m2);
w = m2 - m1;

figure(3)
hold on
plot(m1)
plot(m2)
plot(w)
title('Mean vectors')
grid on; grid minor
xlim([1 220])
legend('Class 1','Class 2','Difference')

% Without PCA
for k = 1:length(classTest)
    cl(k) = sign(w*(classTest(:,k)'-x0)');
    ppp(k) = w*(classTest(:,k)'-x0)';
end
acc = (sum(cl(1:length(class1test)) == -1) + sum(cl(length(class1test)+1:end) == +1))/length(test);
display(['Accuracy without PCA: ', num2str(acc*100), '%'])
figure(10)
subplot(2,1,1)
plot(cl)
title('Without PCA')

figure(11)
subplot(2,1,1)
plot(classTest)
title('Original data')


% With PCA
m1 = m1 - mTrain;
m2 = m2 - mTrain;
x0 = 0.5 * (m1+m2);
w = m2 - m1;

figure(12)
hold on
plot(m1)
plot(m2)
plot(w)
title('Mean vectors in PCA')
grid on; grid minor
xlim([1 220])
legend('Class 1','Class 2','Difference')

K = 50

xHat = xHat - mTrain';
for k = 1:length(xHat)
    cl(k) = sign(w*(xHat(:,k)'-x0)');
    pp(k) = w*(xHat(:,k)'-x0)';
end
figure(99)
plot(pp)
title('pp')
acc = (sum(cl(1:length(class1test)) == -1) + sum(cl(length(class1test)+1:end) == +1))/length(test);
display(['Accuracy with PCA: ', num2str(acc*100), '%'])


figure(10)
subplot(2,1,2)
plot(cl)
title('PCA')

figure(11)
subplot(2,1,2)
plot(xHat)
title('xHat')