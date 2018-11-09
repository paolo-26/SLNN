close all; clear; clc;
data = load('localization.mat');
train=data.traindata;
test=data.testdata;
Nc =24;
accuracy=0;
range=1:120;
o=1;
mat_class = [-1 -1 -1 -1 -1 -1; -1 1 2 3 4 -1; -1 5 6 7 8 -1; -1 9 10 11 12 -1; -1 13 14 15 16 -1; -1 17 18 19 20 -1; -1 21 22 23 24 -1; -1 -1 -1 -1 -1 -1];
for run=1:2
    for k=1:120
        for n = 1:Nc  %colonna di partenza
            for m = 1:Nc 
                dist(m,:,:)=pdist2(train(:,:,n)',train(:,:,m)');
            end
            for m = 1:5
                    riordiniamo=reshape(dist(:,:,m)',120,1);
                    c=1;
                    for i=1:5:120 
                       riordiniamo(i:i+4,2) = c;
                       c=c+1;
                    end
            riordiniamo = sortrows(riordiniamo,1);
            results(m) = mode(riordiniamo(1:k,2));
            end
           accuracy=accuracy+sum(results==n) ;
           if run==2
               [r,c]=find(mat_class==n);
               accuracy = accuracy + sum(mat_class(r-1,c-1)==results);
               accuracy = accuracy + sum(mat_class(r-1,c)==results);
               accuracy = accuracy + sum(mat_class(r-1,c+1)==results);
               accuracy = accuracy + sum(mat_class(r,c-1)==results);
               accuracy = accuracy + sum(mat_class(r,c+1)==results);
               accuracy = accuracy + sum(mat_class(r+1,c-1)==results);
               accuracy = accuracy + sum(mat_class(r+1,c)==results);
               accuracy = accuracy + sum(mat_class(r+1,c+1)==results);
           end
        end
        acc(o)=accuracy/120;
        accuracy=0;
        o=o+1;
    end
    if run == 1
        figure(1)
        hold on
        plot(1-acc,'.-b')
        grid on
        grid minor
        clear acc
        o = 1;
    end
    if run == 2
        plot(1-acc,':ob','markersize',2)
    end
    o=1
    for k=1:120
        for n = 1:Nc  %colonna di partenza
            for m = 1:Nc 
                dist(m,:,:)=pdist2(train(:,:,n)',test(:,:,m)');
            end
            for m = 1:5
                    riordiniamo=reshape(dist(:,:,m)',120,1);
                    c=1;
                    for i=1:5:120 
                       riordiniamo(i:i+4,2) = c;
                       c=c+1;
                    end
            riordiniamo = sortrows(riordiniamo,1);
            results(m) = mode(riordiniamo(1:k,2));
            end
           accuracy=accuracy+sum(results==n) ;
           if run == 2
               [r,c]=find(mat_class==n);
               accuracy = accuracy + sum(mat_class(r-1,c-1)==results);
               accuracy = accuracy + sum(mat_class(r-1,c)==results);
               accuracy = accuracy + sum(mat_class(r-1,c+1)==results);
               accuracy = accuracy + sum(mat_class(r,c-1)==results);
               accuracy = accuracy + sum(mat_class(r,c+1)==results);
               accuracy = accuracy + sum(mat_class(r+1,c-1)==results);
               accuracy = accuracy + sum(mat_class(r+1,c)==results);
               accuracy = accuracy + sum(mat_class(r+1,c+1)==results);
           end
        end
        acc2(o)=accuracy/120;
        accuracy=0;
        o=o+1;
    end
    if run == 1
        plot(1-acc2,'.-r')
        clear acc2
        o = 1;
    end
    if run == 2
        plot(1-acc2,':or','markersize',2)
    end
end
legend('Training set','Test set','Training set (adjacent cells included)','Test set (adjacent cells included)','location','southeast')
xlabel('k nearest neighbours')
ylabel('Misclassification rate')
title('k-NN classifier')


% 
% for n = 1:24
%     for m = 1:24
%         
%         dist()=pdist2(train(:,:,m)',train(:,:,n)');
%         minima=min(dist)
%         celldistance(m)=min(minima)
%         
%     end
%         [value,index] = min(celldistance)
%         results(n)=index
% end