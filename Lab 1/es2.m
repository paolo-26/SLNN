close all; clear; clc;

data = load('speech_dataset.mat');

split=3838
original_dataset = data.dataset(1:split,:);
original_datatest = data.dataset(split+1:end,:);

data_train = original_dataset;
data_train(:,9)=1:length(data_train(:,1));

data_test = original_datatest;
data_test(:,9)=1:length(data_test(:,1));
classes = [1 2]

c=1;
k_range=1:100:split;
k_range'
for k=k_range
    k
    
    for pt=1:1:length(data_train(:,1)) 
    data_train = sortrows(data_train,9);
    data_train(:,7)=pdist2(original_dataset(pt,1:5),data_train(:,1:5),'euclidean');
    data_train = sortrows(data_train,7);      %riordino per distanze
    data_train(1,8)=mode(data_train(1:k,6)); %salva la classe per il punto pt
    end
    accuracy(c) = sum(data_train(:,6)==data_train(:,8))/length(data_train);
    c=c+1;
end

figure(1)
hold on
plot(k_range,1-accuracy,'.-')
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
        data_train = sortrows(data_train,9);
        data_train(:,7)=pdist2(data_test(pt,1:5),original_dataset(:,1:5),'euclidean');
        data_train = sortrows(data_train,7);      %riordino per distanze
        data_test(pt,8)=mode(data_train(1:k,6)); %salva la classe per il punto pt
    end
    accuracy2(c) = sum(data_test(:,6)==data_test(:,8))/length(data_test);
    c=c+1;
end

plot(k_range,1-accuracy2,'.-')
legend('Training set', 'Test set')
%ylim([0 1])
xticks([k_range(1:9:end)])
xlim([0,split+1])