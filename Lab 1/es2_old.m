close all; clear; clc;

data = load('speech_dataset.mat');
original_dataset = data.dataset(1:500,:);
dataset = original_dataset;
dataset(:,9)=1:length(dataset(:,1))
classes = [1 2]
c=1
k_range=[1 10 20 30]
for k=k_range
    k
    for pt=1:1:length(dataset(:,1)) 
        for i=1:1:length(dataset(:,1))
            %pt_coor = ;
            %pti_coor = );
            dataset(i,7) = pdist([original_dataset(pt,1:5); dataset(i,1:5)]);
            
        end
            %=dist;   %salvo la distanza da tutti i punti per il punto pt
            % ;    %aggiungo l'indice dei punti 
            dataset = sortrows(dataset,7);      %riordino per distanze
            
            %occur(:,1) = histc(dataset(1:0+k,6),classes);  %calcolo le occorrenze per ogni classe
            %occur(:,2)=classes;     %aggiungo le classi a occur
            %occur = sortrows(occur,1,'descend'); %ordina le classi per occorrenze
            dataset(1,8)=mode(dataset(1:k,6)); %salva la classe per il punto pt
            %pt
            %dataset = sortrows(dataset,9);
    end
    accuracy(c) = sum(dataset(:,6)==dataset(:,8))/length(dataset);
    c=c+1;
end

figure(1)
plot(k_range,1-accuracy,'d:')
ylabel('Misclassification rate')
xlabel('k')
grid on
grid minor

