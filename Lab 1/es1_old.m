close all; clear; clc;
set=load('synthetic.mat')
%load('speech_dataset.mat')
%load('localization.mat')
train = set.knnClassify2dTrain
test = set.knnClassify2dTest

classes = [1 2]

% scatterplot

figure(1)
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

figure(2)
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
title('Testing set')
legend('class 1', 'class 2','location','southwest')
xlabel('\it x')
ylabel('\it y')
xlim([-8 4])
grid on
grid minor

for k=0:99
    for pt=1:1:length(train(:,1)) 
        for i=1:1:length(train(:,1))
            Xpt=train(pt,1);
            Ypt=train(pt,2);
            dist(i) = pdist2([Xpt Ypt],[train(i,1) train(i,2)]);
            train_pt = train;
        end
            train_pt(:,4)=dist;   %salvo la distanza da tutti i punti per il punto pt
            train_pt(:,5)=1:length(train(:,1)) ;    %aggiungo l'indice dei punti 
            train_pt = sortrows(train_pt,4);      %riordino per distanze
            occur(:,1) = histc(train_pt(1:1+k,3),classes);  %calcolo le occorrenze per ogni classe
            occur(:,2)=classes;     %aggiungo le classi a occur
            occur = sortrows(occur,1,'descend'); %ordina le classi per occorrenze
            train(pt,4)=occur(1,2); %salva la classe per il punto pt
    end
    accuracy(k+1) = sum(train(:,3)==train(:,4));
end

figure(3)
hold on
plot([1:100],(length(test(:,1))-accuracy)/(length(test(:,1))),'d-','MarkerFaceColor','c')
grid on
grid minor

clear occur
clear accuracy


for k=0:99
    for pt=1:1:length(test(:,1)) 
        for i=1:1:length(test(:,1))
            Xpt=test(pt,1);
            Ypt=test(pt,2);
            dist(i) = pdist2([Xpt Ypt],[train(i,1) train(i,2)]);
            test_pt = train;
        end
            test_pt(:,4)=dist;   %salvo la distanza da tutti i punti per il punto pt
            test_pt(:,5)=1:length(test(:,1)) ;    %aggiungo l'indice dei punti 
            test_pt = sortrows(test_pt,4);      %riordino per distanze
            occur(:,1) = histc(test_pt(1:1+k,3),classes);  %calcolo le occorrenze per ogni classe
            occur(:,2)=classes;     %aggiungo le classi a occur
            occur = sortrows(occur,1,'descend'); %ordina le classi per occorrenze
            test(pt,4)=occur(1,2); %salva la classe per il punto pt
    end
    accuracy(k+1) = sum(test(:,3)==test(:,4));
end

plot([1:100],(length(test(:,1))-accuracy)/(length(test(:,1))),'d-','markerfacecolor','m')
legend('train','test','location','best')
title('Misclassification rate')
xlabel('K')



