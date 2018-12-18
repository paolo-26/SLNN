clc; clear; close all;
cartadazucchero = [137; 207; 240]/255;
ametista = [153; 102; 204]/255;
data = load('XwindowsDocData.mat');
err = 0.05

train = [sum(data.ytrain == 1) sum(data.ytrain == 2)];
test = [sum(data.ytest == 1) sum(data.ytest == 2)];

theta(:,1) = sum(data.xtrain(1:train(1),:) == 1)/train(1);
theta(:,2) = sum(data.xtrain(train(2)+1:end,:) == 1)/train(2);
pie(1) = train(1)/length(data.ytrain);
pie(2) = train(2)/length(data.ytrain);

figure(1)
subplot(2,1,1)
bar(theta(1:end,1),'k')
title('Microsoft Windows')
grid minor
subplot(2,1,2)
bar(theta(1:end,2),'k')
title('Windows X')
grid minor

figure(2)
hold on
stem(theta(1:end,1), 'marker','o', 'color',cartadazucchero, 'markersize',4)
stem(theta(1:end,2), 'marker','^', 'markersize',4)
%uninformative = (theta(1:end,1) == theta(1:end,2));
uninformativeWords = (abs(theta(1:end,1)-theta(1:end,2))) <= err;  % error
%plot(find(unin2),theta(find(unin2)),'kx','markerfacecolor','k','markersize',10)
grid on
legend('Microsoft Windows', 'X Windows', 'location','best')
title('All features')

figure(3)
hold on
stem(find(uninformativeWords == 0),theta(find(uninformativeWords == 0),1),'marker','o', 'color','red', 'markersize',4)
stem(find(uninformativeWords == 0),theta(find(uninformativeWords == 0),2),'marker','^', 'markersize',4)
grid on
title(['Features that differs by at most ', num2str(err)])

%plot(find(unin2),find(unin2),'.')
%xticks([find(unin)])
%xticklabels(data.vocab(find(unin)))
%xtickangle(90)
legend('Microsoft Windows','X Windows', 'location','best')

% subplots
% figure(10)
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
%xtickangle(90)
