close all; clear; clc;

data = load('speech_dataset.mat');

split=500
original_dataset = data.dataset(1:split,:);
original_datatest = data.dataset(split+1:split*2,:);

data_train = original_dataset;
data_train(:,9)=1:length(data_train(:,1));

data_test = original_datatest;
data_test(:,9)=1:length(data_test(:,1));
classes = [1 2]

c=1;
k_range=1:25:100
for k=k_range
    k
    for pt=1:1:length(data_train(:,1)) 
        for i=1:1:length(data_train(:,1))
            data_train(i,7) = pdist([original_dataset(pt,1:5); data_train(i,1:5)]);
        end
            data_train = sortrows(data_train,7);      %riordino per distanze
            data_train(1,8)=mode(data_train(1:k,6)); %salva la classe per il punto pt
    end
    accuracy(c) = sum(data_train(:,6)==data_train(:,8))/length(data_train);
    c=c+1;
end

figure(1)
hold on
plot(k_range,1-accuracy,'d-','MarkerFaceColor','c')
ylabel('Misclassification rate')
xlabel('Number of neighbors \itk')
grid on
grid minor


data_train = original_dataset;
data_train(:,9)=1:length(data_train(:,1));

c=1;
for k=k_range
    k
    for pt=1:1:length(data_test(:,1))  %per tutti i punti del data_test
        for i=1:1:length(data_train(:,1)) %per tutti i punti del data_train
            data_train(i,7) = pdist([original_dataset(i,1:5); data_test(pt,1:5)]);
        end
            data_train = sortrows(data_train,7);      %riordino per distanze
            data_test(pt,8)=mode(data_train(1:k,6)); %salva la classe per il punto pt
    end
    accuracy2(c) = sum(data_test(:,6)==data_test(:,8))/length(data_test);
    c=c+1;
end

plot(k_range,1-accuracy2,'d-','MarkerFaceColor','m')
legend('Training set', 'Testing  set')
ylim([0 1])
xticks(k_range)