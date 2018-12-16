clear; clc; close all;

data = load('Indian_Pines_Dataset')
indian_pines = data.indian_pines;
indian_pines_gt = data.indian_pines_gt;

C1 = 830;  % Number of samples of class 1
C2 = 386;  % Number of samples of class 2
t1 = 3  % Index of class 1
t2 = 15  % Index of class 2
N_SPECTR = 220;  % Number of features
SPLIT = 0.75  % 75% trainining and 25% test
SPLIT1 = round(C1*SPLIT)  % Split training-test for class 1
SPLIT2 = round(C2*SPLIT)  % Split training-test for class 2


%% Extract data
n=0;
class1 = zeros(C1, N_SPECTR);
for i = 1:size(indian_pines, 1)
    for j = 1:size(indian_pines, 2)
        if indian_pines_gt(i,j)== t1 % class index
            n = n + 1;
            class1(n,:) = indian_pines(i,j,:);
        end
    end
end

n = 0;
class2 = zeros(C2, N_SPECTR);
for i = 1:size(indian_pines, 1)
    for j = 1:size(indian_pines, 2)
        if indian_pines_gt(i,j)== t2 % class index
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
classTrain = [class1train; class2train]';  % Test set with class 1 and 2
classTest = [class1test; class2test]';  % Training set with class 1 and 2
mTrain = mean(classTrain, 2);  % Mean for both classes
m1 = mean(class1train)';  % Mean for class1
m2 = mean(class2train)';  % Mean for class2
%class1test = class1test - m1;  % Centering the data (column means)
%class2test = class2test - m2;  % Centering the data (column means)
% figure(9)
% hold on
% plot(x)
% covariance2 = 1/N * x(:,:) * x(:,:)';
% covariance3 = cov([class1train; class2train]);

%% Covariance matrix on training set
N = SPLIT1 + SPLIT2;
cov_mat = 0;
x = [class1train' class2train']-mTrain;
for k = 1:SPLIT1+SPLIT2
    cov_mat = cov_mat + (x(:,k)*x(:,k)')/N;
end

%% PCA
[V,D] = eig(cov_mat);  % Eigendecomposition

o = 1;
test = classTest - mTrain;  % Zero-mean data for PCA

%xHat = sqrt(inv(Lambda))*W'*xHat;
for K=1:220
    Lambda = D(end-K+1:end, end-K+1:end);
    W = V(:,end-K+1:end);
    z = W'*test;
    xHat = W*z + mTrain;  % Add mean to reflect original data
    MSE(o) = (norm(xHat - (classTest)))^2/length(test);
    o = o + 1;    
end

figure(1)
hold on
semilogy(MSE)
grid on
title('Mean squared error on test set')
legend('MSE (test)')
grid minor
xlim([1 220])
xlabel('K')
ylabel('MSE')

% Find most important eigenvectors
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
% w = m2 - m1;
w = inv(cov_mat)*m2 - m1;

figure(3)
hold on
plot(m1)
plot(m2)
plot(w)
title('Mean vectors')
grid on; grid minor
xlim([1 220])
legend('Class 1','Class 2','Difference')

% 1. Without PCA on the original data
for k = 1:length(classTest)
    cl(k) = sign(w'*(classTest(:,k)-x0));
%     ppp(k) = w'*(classTest(:,k)-x0);
end
acc = (sum(cl(1:length(class1test)) == -1) + sum(cl(length(class1test)+1:end) == +1))/length(test);
display(['Classes: ', num2str(t1), ' ', num2str(t2)])
display(['Accuracy without PCA: ', num2str(acc*100), '%'])

% figure(98)
% plot(ppp)
% title('no PCA')

% figure(10)
% subplot(2,1,1)
% plot(cl)
% title('Without PCA')
% figure(11)
% subplot(2,1,1)
% plot(classTest)
% title('Original data')


% 3. With only N retained features
N = 100;  % Retained features
for k = 1:length(classTest)
    cl(k) = sign(w(1:N)'*(classTest(1:N,k)-x0(1:N)));
end
acc = (sum(cl(1:length(class1test)) == -1) + sum(cl(length(class1test)+1:end) == +1))/length(test);
display(['Accuracy with ', num2str(N), ' retained features: ', num2str(acc*100), '%'])
    

% 2. With PCA + whitening
train = classTrain - mTrain;
for K = 1:220
    % PCA + whitening on test set
    Lambda = D(end-K+1:end, end-K+1:end);
    W = V(:,end-K+1:end);
    z = sqrt(inv(Lambda))*W'*test;  % Whitening
    xHat = W*z + mTrain;
    
    % PCA + whitening on training set
    Lambda = D(end-K+1:end, end-K+1:end);
    W = V(:,end-K+1:end);
    z = sqrt(inv(Lambda))*W'*train;  % Whitening
    xHat_train = W*z + mTrain;

    % New mean vectors for class1 and class2 on the new training set
    xHat = xHat - mTrain;
    xHat_train = xHat_train - mTrain;
    m1 = mean(xHat_train(:,1:SPLIT1), 2);
    m2 = mean(xHat_train(:,SPLIT1+1:end), 2);
    
    sigma=cov(z);
    x0 = 0.5 * (m1+m2);
    w = m2 - m1;
    
    for k = 1:length(xHat)
        cl(k) = sign(w'*(xHat(:,k)-x0));
%         pp(k) = w'*(xHat(:,k)-x0);
    end
    acc(K) = (sum(cl(1:length(class1test)) == -1) + sum(cl(length(class1test)+1:end) == +1))/length(test);
    
%     figure(12)
%     hold on
%     plot(m1)
%     plot(m2)
%     plot(w)
%     title('Mean vectors in PCA')
%     grid on; grid minor
%     xlim([1 220])
%     legend('Class 1','Class 2','Difference')

%     figure(99)
%     plot(pp)
%     title('PCA')

%     figure(10)
%     subplot(2,1,2)
%     plot(cl)
%     title('PCA')
% 
end
% figure(11)
% subplot(2,1,2)
% plot(xHat)
% title('xHat')
     
figure(4)
plot(acc)
title('Accuracy with PCA + whitening')
xlabel('K')
ylabel('Accuracy')
xlim([0,220])
grid minor
