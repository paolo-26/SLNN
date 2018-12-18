close all; clear; clc;

data = load('speech_dataset.mat');

split=2000  % Split training and test set.
rangeK = 1:50:split;

c1=(data.dataset(:,6)==1);
c2=(data.dataset(:,6)==2);

o = 1
p = 1
for i = 1:length(data.dataset)
    if data.dataset(i,6) == 1;
       class1(o,:) = data.dataset(i,:);
       o = o + 1;
    else
       class2(p,:) = data.dataset(i,:);
       p = p + 1;
    end
end

originalDataTrain = [class1(1:round(0.7*length(class1)),:);
                     class2(1:round(0.7*length(class2)),:)];
originalDataTest = [class1(round(0.7*length(class1))+1:end,:);
                     class2(round(0.7*length(class2))+1:end,:)];
                 
                 
% originalDataTrain = data.dataset(1:split,:);
% originalDataTest = data.dataset(split+1:end,:);

dataTrain = originalDataTrain;
dataTrain(:,9) = 1:length(dataTrain(:,1));

dataTest = originalDataTest;
dataTest(:,9) = 1:length(dataTest(:,1));
classes = [1 2]

c = 1;

rangeK'
for k = rangeK
    k  % Print the current k to track the progress.
    
    for pt = 1:1:length(dataTrain(:,1)) 
        dataTrain = sortrows(dataTrain,9);
        dataTrain(:,7) = pdist2(originalDataTrain(pt,1:5),dataTrain(:,1:5),'euclidean');
        dataTrain = sortrows(dataTrain,7);  % Reorder the distances.
        dataTrain(1,8) = mode(dataTrain(1:k,6)); % Save the class for point pt.
    end
    accuracy(c) = sum(dataTrain(:,6) == dataTrain(:,8))/length(dataTrain);
    c=c+1;
end

figure(1)
hold on
plot(rangeK,1-accuracy,'.-')
ylabel('Misclassification rate')
xlabel('Number of neighbors \itk')
grid on
grid minor


dataTrain = originalDataTrain;
dataTrain(:,9) = 1:length(dataTrain(:,1));

c=1;
for k=rangeK
    k  % Print the current k to track the progress.
    for pt=1:1:length(dataTest(:,1))
        dataTrain = sortrows(dataTrain,9);
        dataTrain(:,7) = pdist2(dataTest(pt,1:5),originalDataTrain(:,1:5),'euclidean');
        dataTrain = sortrows(dataTrain,7);  % Reorder the distances.
        dataTest(pt,8) = mode(dataTrain(1:k,6));  % Save the class for point pt.
    end
    accuracy2(c) = sum(dataTest(:,6) == dataTest(:,8))/length(dataTest);
    c=c+1;
end

plot(rangeK,1 - accuracy2,'.-')
legend('Training set', 'Test set', 'location', 'southeast')
%ylim([0 1])
xticks([rangeK(1:9:end)])
xlim([0,split + 1])