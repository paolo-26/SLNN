clear; clc; close all;
data = load('localization.mat');
train = data.traindata;
test = data.testdata;
N_CELL = size(data.testdata,3);
rangeK = 1:120;
matrixClass = [-1 -1 -1 -1 -1 -1; -1 1 2 3 4 -1; -1 5 6 7 8 -1;...
               -1 9 10 11 12 -1; -1 13 14 15 16 -1; -1 17 18 19 20 -1;...
               -1 21 22 23 24 -1; -1 -1 -1 -1 -1 -1];

for run = 1:1
    trainError = zeros(1,120);
    for k = rangeK
        accuracy = zeros(1,24);
        for c = 1:24
           class = zeros(1,5);
           for m = 1:5
                dist = zeros(24,5);
                for C = 1:24
                    for M = 1:5
                        % misurazione m della cella c
                        % confrontato con tutti i vettori M di tutte le celle C
                        dist(C,M) = norm(train(:,m,c)-train(:,M,C));
                    end
                end
                % trovo i k vettori pi� vicini:
                d = reshape(dist',1,120);
                d(2,1:120) = repelem(1:24,5);
                d = d';
                d = sortrows(d,1);
                class(m) = mode(d(1:k,2));
           end
           %accuracy = accuracy + sum(class == c);
           if run == 2
               [row, col] = find(matrixClass==c);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row-1,col-1) == class);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row-1,col) == class);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row-1,col+1) == class);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row,col-1) == class);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row,col+1) == class);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row+1,col-1) == class);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row+1,col) == class);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row+1,col+1) == class);
           end
           accuracy(c) = (accuracy(c) + sum(class == c))/5;
        end
        %trainError(k) = 1 - accuracy/120;
        trainError(k) = 1 - mean(accuracy);
    end

    testError = zeros(1,120);
    for k = rangeK
        accuracy = zeros(1,24);
        for c = 1:24
           class = zeros(1,5);
           for m = 1:5
                dist = zeros(24,5);
                for C = 1:24
                    for M = 1:5
                        % misurazione m della cella c
                        % confrontato con tutti i vettori M di tutte le celle C
                        dist(C,M) = norm(test(:,m,c)-train(:,M,C));
                    end
                end
                % trovo i k vettori pi� vicini:
                d = reshape(dist',1,120);
                d(2,1:120) = repelem(1:24,5);
                d = d';
                d = sortrows(d,1);
                class(m) = mode(d(1:k,2));
           end
           %accuracy = accuracy + sum(class == c);
           if run == 2
               [row, col] = find(matrixClass==c);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row-1,col-1) == class);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row-1,col) == class);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row-1,col+1) == class);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row,col-1) == class);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row,col+1) == class);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row+1,col-1) == class);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row+1,col) == class);
               accuracy(c) = accuracy(c) +...
                             sum(matrixClass(row+1,col+1) == class);
           end
           accuracy(c) = (accuracy(c) + sum(class == c))/5;
        end
        %testError(k) = 1 - accuracy/120;
        testError(k) = 1 - mean(accuracy);

    end

    figure(1)
    hold on
    grid on
    grid minor
    plot(trainError, '.-')
    plot(testError, '.-')

end

legend('Training set', 'Test set', 'location','southeast')
xlabel('Number of neighbours k')
ylabel('Misclassification rate')