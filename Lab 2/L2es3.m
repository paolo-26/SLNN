clc; clear; close all;
cartadazucchero = [137; 207; 240]/255;
ametista = [153; 102; 204]/255;
data = load('XwindowsDocData.mat')
err = 0.05

train = [sum(data.ytrain == 1) sum(data.ytrain == 2)];
test = [sum(data.ytest == 1) sum(data.ytest == 2)];

theta(:,1) = sum(data.xtrain(1:train(1),:) == 1)/train(1);
theta(:,2) = sum(data.xtrain(train(2)+1:end,:) == 1)/train(2);
pie(1) = train(1)/length(data.ytrain);
pie(2) = train(2)/length(data.ytrain);

% figure(1)
% subplot(2,1,1)
% bar(theta(1:end,1),'k')
% title('Microsoft Windows')
% grid minor
% subplot(2,1,2)
% bar(theta(1:end,2),'k')
% title('Windows X')
% grid minor
% 
% figure(2)
% hold on
% stem(theta(1:end,1), 'marker','o', 'color',cartadazucchero, 'markersize',4)
% stem(theta(1:end,2), 'marker','^', 'markersize',4)
% %uninformative = (theta(1:end,1) == theta(1:end,2));
% uninformativeWords = (abs(theta(1:end,1)-theta(1:end,2))) <= err;  % error
% %plot(find(unin2),theta(find(unin2)),'kx','markerfacecolor','k','markersize',10)
% grid on
% legend('Microsoft Windows', 'X Windows', 'location','best')
% title('All features')
% 
% figure(3)
% hold on
% stem(find(uninformativeWords == 0),theta(find(uninformativeWords == 0),1),'marker','o', 'color','red', 'markersize',4)
% stem(find(uninformativeWords == 0),theta(find(uninformativeWords == 0),2),'marker','^', 'markersize',4)
% grid on
% title(['Features that differs by at most ', num2str(err)])

%plot(find(unin2),find(unin2),'.')
%xticks([find(unin)])
%xticklabels(data.vocab(find(unin)))
%xtickangle(90)
%legend('Microsoft Windows','X Windows', 'location','best')


%% esercizio 3

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

acc(1) = sum(classesTrain == data.ytrain)/length(data.ytrain)*100;
acc(2) = sum(classesTest == data.ytest)/length(data.ytest)*100;

%% Optional part

theta=full(theta); 
for j = 1:length(theta)
    thetaJ = sum(pie(1,:).*theta(j,:));
    I(j) = sum(theta(j,:).*pie(1,:).*log((theta(j,:)+eps)/(thetaJ+eps))+(1-theta(j,:)).*pie(1,:).*log((1-theta(j,:)+eps)/(1-thetaJ+eps)));
end
I=I';
I(:,2)=1:length(I);
I=sortrows(I,1,'descend');

o = 1;
for K = 20:20  % Accuracy for each K
features=I(1:K,2);
    
for k = 1:length(data.xtrain)
    resTrain(k,:) = sum(log(theta(find(data.xtrain(k,features)==1),:))) +...
        sum(log(1-theta(find(data.xtrain(k,features)==0),:))) + log(pie(1));
end

for k = 1:length(data.xtest)
    resTest(k,:) = sum(log(theta(find(data.xtest(k,features)==1),:))) +...
        sum(log(1-theta(find(data.xtest(k,features)==0),:))) + log(pie(1));
end

classesTrain = (resTrain(:,1) < resTrain(:,2))+1;
classesTest = (resTest(:,1) < resTest(:,2))+1;

acc(o,1) = sum(classesTrain == data.ytrain)/length(data.ytrain)*100;
acc(o,2) = sum(classesTest == data.ytest)/length(data.ytest)*100;
o = o + 1;
end

figure()
hold on
plot(acc(:,1)/100)
plot(acc(:,2)/100)
legend('Train', 'Test')
grid on
xlabel('K')
ylabel('Accuracy')
ylim([0 1])


