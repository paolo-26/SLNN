clc; clear; close all;
cartadazucchero = [137; 207; 240]/255;
ametista = [153; 102; 204]/255;
data=load('XwindowsDocData.mat')

train=[sum(data.ytrain==1) sum(data.ytrain==2)];
test=[sum(data.ytest==1) sum(data.ytest==2)];

theta(:,1)=sum(data.xtrain(1:train(1),:)==1)/train(1);
theta(:,2)=sum(data.xtrain(train(1)+1:end,:)==1)/train(1);

figure(1)
hold on
stem(theta(1:end,1),'marker','o','color',cartadazucchero,'markersize',4)
stem(theta(1:end,2),'marker','^','markersize',4)
unin=(theta(1:end,1)==theta(1:end,2));
plot(find(unin),theta(find(unin)),'kx','markerfacecolor','k','markersize',10)
grid on
%xticks([find(unin)])
%xticklabels(data.vocab(find(unin)))
%xtickangle(90)
legend('Microsoft Windows','X Windows')


% figure(2)
% subplot(2,1,1)
% hold on
% plot(theta(1:end/2,1),'marker','o','color',cartadazucchero,'markersize',4)
% plot(theta(1:end/2,2),'marker','^','markersize',4)
% unin=(theta(1:end/2,1)==theta(1:end/2,2))
% plot(find(unin),theta(find(unin)),'kx','markerfacecolor','k','markersize',10)
% 
% subplot(2,1,2)
% hold on
% plot(theta(end/2+1:end,1),'marker','o','color',cartadazucchero,'markersize',4)
% plot(theta(end/2+1:end,2),'marker','^','markersize',4)
% unin=(theta(end/2+1:end,1)==theta(end/2+1:end,2))
% plot(find(unin),theta(find(unin)),'kx','markerfacecolor','k','markersize',10)
% %xtickangle(90)
