clc; clear; close all;
data = load('XwindowsDocData.mat');

train = [sum(data.ytrain == 1) sum(data.ytrain == 2)];
test = [sum(data.ytest == 1) sum(data.ytest == 2)];

theta(:,1) = sum(data.xtrain(1:train(1),:) == 1)/train(1);
theta(:,2) = sum(data.xtrain(train(2)+1:end,:) == 1)/train(2);
pie = [train(1)/length(data.ytrain);
       train(2)/length(data.ytrain)];

   theta = full(theta);
for k = 1:length(data.xtrain)
    resTrain(k,:) = sum(log(theta(find(data.xtrain(k,:)==1),:))) +...
        sum(log(1-theta(find(data.xtrain(k,:)==0),:))) + log(pie(1));
end

for k = 1:length(data.xtest)
    resTest(k,:) = sum(log(theta(find(data.xtest(k,:)==1),:))) +...
        sum(log(1-theta(find(data.xtest(k,:)==0),:))) + log(pie(1));
end

classesTrain = (resTrain(:,1) < resTrain(:,2))+1;
classesTest = (resTest(:,1) < resTest(:,2))+1;

acc(1) = sum(classesTrain == data.ytrain)/length(data.ytrain)*100
acc(2) = sum(classesTest == data.ytest)/length(data.ytest)*100

%% Optional part

% for j = 1:length(theta)
%     thetaJ = sum(pie.*theta(j,:)');
%     I(j) = sum(theta(j,:).*pie(1,:).*log((theta(j,:)+eps)/(thetaJ+eps))+...
%         (1-theta(j,:)).*pie(1,:).*log((1-theta(j,:)+eps)/(1-thetaJ+eps)));
% end

I = zeros(1,600);


for j = 1:length(theta)
   thetaJ = sum(pie.*theta(j,:)');
   thetaJ2(j) = thetaJ;
   for class=1:2
    I(j) = I(j) + theta(j,class)*pie(class,1)*log((theta(j,class)+eps)/(thetaJ+eps))+(1-theta(j,class))*pie(class,1)*log((1-theta(j,class)+eps)/(1-thetaJ+eps));
   end
end

I = I';  % Make I a column vector
I(:,2) = 1:length(I);  % Add indexing to I
I = sortrows(I,1,'descend');  % Sort: most important words

o = 1;

for K = 1:600  % Accuracy for each K
features = I(1:K,2);  % Most important words

for k = 1:length(data.xtrain)
    resTrain(k,1) = sum(log(theta(features(data.xtrain(k,features)==1),1))) +...
        sum(log(1-theta(features(data.xtrain(k,features)==0),1)));% + log(pie(1));
end

for k = 1:length(data.xtrain)
    resTrain(k,2) = sum(log(theta(features(data.xtrain(k,features)==1),2))) +...
        sum(log(1-theta(features(data.xtrain(k,features)==0),2)));% + log(pie(1));
end

for k = 1:length(data.xtest)
    resTest(k,1) = sum(log(theta(features(data.xtest(k,features)==1),1))) +...
        sum(log(1-theta(features(data.xtest(k,features)==0),1)));% + log(pie(1));
end

for k = 1:length(data.xtest)
    resTest(k,2) = sum(log(theta(features(data.xtest(k,features)==1),2))) +...
        sum(log(1-theta(features(data.xtest(k,features)==0),2)));% + log(pie(1));
end


classesTrain = (resTrain(:,1) < resTrain(:,2)) + 1;  % Choose class
classesTest = (resTest(:,1) < resTest(:,2)) + 1;  % Choose class
acc(o,1) = sum(classesTrain == data.ytrain) / length(data.ytrain) * 100;
acc(o,2) = sum(classesTest == data.ytest) / length(data.ytest) * 100;
o = o + 1;
end

figure(1)
hold on
plot(acc(:,1)/100,'.-')
plot(acc(:,2)/100,'.-')
legend('Training', 'Test', 'location','best')
title('Naive Bayes classifier')
grid on
grid minor
xlabel('K')
ylabel('Accuracy')
ylim([0 1])