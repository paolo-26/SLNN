close all; clear; clc;

data = load('synthetic.mat');


original_dataset = data.knnClassify2dTrain;
original_datatest = data.knnClassify2dTest;

train = data.knnClassify2dTrain;
test = data.knnClassify2dTest;

data_train = original_dataset;
data_train(:,6)=1:length(data_train(:,1));

data_test = original_datatest;
data_test(:,6)=1:length(data_test(:,1));
classes = [1 2]

c=1;
k_range=1:100
for k=k_range
    k;
    
    for pt=1:1:length(data_train(:,1)) 
    data_train = sortrows(data_train,6);    
    data_train(:,4)=pdist2(original_dataset(pt,1:2),data_train(:,1:2),'euclidean');
        %for i=1:1:length(data_train(:,1))
         %  data_train(i,7) = pdist([original_dataset(pt,1:5); data_train(i,1:5)]);
        %end
            data_train = sortrows(data_train,4);      %riordino per distanze
            data_train(1,5)=mode(data_train(1:k,3)); %salva la classe per il punto pt
    end
    accuracy(c) = sum(data_train(:,3)==data_train(:,5))/length(data_train);
    c=c+1;
end

figure(1)
hold on
plot(k_range,1-accuracy,'o-','MarkerFaceColor','c')
ylabel('Misclassification rate')
xlabel('Number of neighbors \itk')
grid on
grid minor


data_train = original_dataset;
data_train(:,6)=1:length(data_train(:,1));

c=1;
for k=k_range
    k;
    for pt=1:1:length(data_test(:,1))  %per tutti i punti del data_test
        data_train = sortrows(data_train,6);
        data_train(:,4)=pdist2(data_test(pt,1:2),original_dataset(:,1:2),'euclidean');
        %for i=1:1:length(data_train(:,1)) %per tutti i punti del data_train
         %   data_train(i,7) = pdist([original_dataset(i,1:5); data_test(pt,1:5)]);
        %end
            data_train = sortrows(data_train,4);      %riordino per distanze
            data_test(pt,5)=mode(data_train(1:k,3)); %salva la classe per il punto pt
    end
    accuracy2(c) = sum(data_test(:,3)==data_test(:,5))/length(data_test);
    c=c+1;
end

plot(k_range,1-accuracy2,'o-','MarkerFaceColor','m')
legend('Training set', 'Test set')
ylim([0 1])
xticks(k_range)



% scatterplot

figure(2)
hold on
class1=sortrows(train,3);
class2=sortrows(train,3);
for c=1:length(class1(:,1))
    if class1(c,3)~=1
        break
    end
end
class1=train;
class1(c:end,:) = []

class2=train;
class2(1:c,:) = []

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
class1=sortrows(test,3);
class2=sortrows(test,3);
for c=1:length(class1(:,1))
    if class1(c,3)~=1
        break
    end
end
class1=test;
class1(c:end,:) = []

class2=test;
class2(1:c,:) = []

scatter(class1(:,1),class1(:,2),'*')
scatter(class2(:,1),class2(:,2),'*')
title('Test set')
legend('class 1', 'class 2','location','southwest')
xlabel('\it x')
ylabel('\it y')
xlim([-8 4])
grid on
grid minor