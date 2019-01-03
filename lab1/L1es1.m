close all; clear; clc;

data = load('synthetic.mat');


originalDataTrain = data.knnClassify2dTrain;
originalDataTest = data.knnClassify2dTest;

train = data.knnClassify2dTrain;
test = data.knnClassify2dTest;

dataTrain = originalDataTrain;
dataTrain(:,6)=1:length(dataTrain(:,1));

dataTest = originalDataTest;
dataTest(:,6)=1:length(dataTest(:,1));
classes = [1 2]

c=1;
rangeK=1:100
for k=rangeK
    k;
    
    for pt=1:1:length(dataTrain(:,1)) 
    dataTrain = sortrows(dataTrain,6);    
    dataTrain(:,4)=pdist2(originalDataTrain(pt,1:2),dataTrain(:,1:2),'euclidean');
        %for i=1:1:length(data_train(:,1))
         %  data_train(i,7) = pdist([original_dataset(pt,1:5); data_train(i,1:5)]);
        %end
            dataTrain = sortrows(dataTrain,4);      %riordino per distanze
            dataTrain(1,5) = mode(dataTrain(1:k,3)); %salva la classe per il punto pt
    end
    accuracy(c) = sum(dataTrain(:,3) == dataTrain(:,5))/length(dataTrain);
    c=c+1;
end

figure(1)
hold on
plot(rangeK,1-accuracy,'o-','MarkerFaceColor','c', 'markersize', 2)
ylabel('Misclassification rate')
xlabel('Number of neighbors \itk')
grid on
grid minor


dataTrain = originalDataTrain;
dataTrain(:,6) = 1:length(dataTrain(:,1));

c=1;
for k=rangeK
    k;
    for pt=1:1:length(dataTest(:,1))  %per tutti i punti del data_test
        dataTrain = sortrows(dataTrain,6);
        dataTrain(:,4) = pdist2(dataTest(pt,1:2),originalDataTrain(:,1:2),'euclidean');
        %for i=1:1:length(data_train(:,1)) %per tutti i punti del data_train
         %   data_train(i,7) = pdist([original_dataset(i,1:5); data_test(pt,1:5)]);
        %end
            dataTrain = sortrows(dataTrain,4);      %riordino per distanze
            dataTest(pt,5) = mode(dataTrain(1:k,3)); %salva la classe per il punto pt
    end
    accuracy2(c) = sum(dataTest(:,3) == dataTest(:,5))/length(dataTest);
    c=c+1;
end

plot(rangeK,1-accuracy2,'o-','MarkerFaceColor','m', 'markersize', 2)
legend('Training set', 'Test set')
%ylim([0 1])
%xticks(rangeK)



% scatterplot

figure(2)
hold on
class1 = sortrows(train,3);
class2 = sortrows(train,3);
for c=1:length(class1(:,1))
    if class1(c,3) ~= 1
        break
    end
end
class1 = train;
class1(c:end,:) = [];

class2 = train;
class2(1:c,:) = [];

scatter(class1(:,1),class1(:,2),'*')
scatter(class2(:,1),class2(:,2),'*')
title('Training set')
legend('class 1', 'class 2','location','southwest')
xlabel('\it x')
ylabel('\it y')
xlim([-8 4])
grid on
grid minor

figure(3)
hold on
class1 = sortrows(test,3);
class2 = sortrows(test,3);
for c=1:length(class1(:,1))
    if class1(c,3)~=1
        break
    end
end
class1 = test;
class1(c:end,:) = [];

class2 = test;
class2(1:c-1,:) = [];

scatter(class1(:,1),class1(:,2),'*')
scatter(class2(:,1),class2(:,2),'*')
title('Test set')
legend('class 1', 'class 2','location','southwest')
xlabel('\it x')
ylabel('\it y')
xlim([-8 4])
grid on
grid minor