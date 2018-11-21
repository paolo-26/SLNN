clear; close all; clc

data = load('heightWeight')
SPLIT_M = 25
SPLIT_F = 40

males = data.heightWeightData(data.heightWeightData(:,1) == 1,2:end);
females = data.heightWeightData(data.heightWeightData(:,1) == 2,2:end);

LM = length(males);
LF = length(females);

testMales = males(1:SPLIT_M,:);
trainMales = males(SPLIT_M+1:end,:);

testFemales = females(1:SPLIT_F,:);
trainFemales = females(SPLIT_F+1:end,:);

for run = 1:3  % Run three times with different covariance matrices
% MLE mean (males).
mM = 0;

for i = 1:length(trainMales)
    mM = mM + trainMales(i,:);
end

mM = mM/length(trainMales)

% MLE mean (females).
mF = 0;

for i = 1:length(trainFemales)
    mF = mF + trainFemales(i,:);
end

mF = mF/length(trainFemales)

% MLE covariance (males).
firstTerm = zeros(2);  % 2x2 matrix of zeros

for i = 1:length(trainMales)
    firstTerm = firstTerm + trainMales(i,:)'*trainMales(i,:); 
end

firstTerm = firstTerm/length(trainMales);
secondTerm = mM.*mM';
sM = firstTerm - secondTerm

if run == 2
   sM = diag(diag(sM))  % Set to zero off-diagonal elements.
end

if run == 3
    firstTerm = zeros(2);  % 2x2 matrix of zeros
    shared = [trainMales; trainFemales];
    for i = 1:length(shared)
        firstTerm = firstTerm + shared(i,:)'*shared(i,:); 
    end
    firstTerm = firstTerm/length(shared);
    secondTerm = mM.*mM';
    sM = firstTerm - secondTerm
end

% MLE covariance (females).
firstTerm = zeros(2);  % 2x2 matrix of zeros
for i = 1:length(trainFemales)
    firstTerm = firstTerm + trainFemales(i,:)'*trainFemales(i,:); 
end
firstTerm = firstTerm/length(trainFemales);
secondTerm = mF.*mF';
sF = firstTerm - secondTerm

if run == 2
   sF = diag(diag(sF))  % Set to zero off-diagonal elements.
end

if run == 3
   sf = sM  % Covariance matrices are the same shared covarianca matrix.
end

pie(1) = [(LM-SPLIT_M)/((LM-SPLIT_M)+(LF-SPLIT_F))];  %  Males
pie(2) = [(LF-SPLIT_F)/((LM-SPLIT_M)+(LF-SPLIT_F))];  %  Females

%% PART 1

x = [testMales; testFemales];
for i = 1:length(x)
    num = pie(1)*(norm(2*pi*sM)^(-1/2))*(exp(-1/2*(x(i,:)-mM)*inv(sM)*(x(i,:)-mM)'));
    den1 = pie(1)*(norm(2*pi*sM)^(-1/2))*(exp(-1/2*(x(i,:)-mM)*inv(sM)*(x(i,:)-mM)'));
    den2 = pie(2)*(norm(2*pi*sF)^(-1/2))*(exp(-1/2*(x(i,:)-mF)*inv(sF)*(x(i,:)-mF)'));
    posteriorM(i) = num/(den1+den2);  % Prob. of being male
end

x = [testMales; testFemales];
for i = 1:length(x)
    num = pie(2)*(norm(2*pi*sF)^(-1/2))*(exp(-1/2*(x(i,:)-mF)*inv(sF)*(x(i,:)-mF)'));
    den1 = pie(1)*(norm(2*pi*sM)^(-1/2))*(exp(-1/2*(x(i,:)-mM)*inv(sM)*(x(i,:)-mM)'));
    den2 = pie(2)*(norm(2*pi*sF)^(-1/2))*(exp(-1/2*(x(i,:)-mF)*inv(sF)*(x(i,:)-mF)'));
    posteriorF(i) = num/(den1+den2);  % Prob. of being female
end
% 
% x = testFemales;
% for i = 1:length(x)
%     num = pie(1)*(norm(2*pi*sF)^(-1/2))*(exp(-1/2*(x(i,:)-mF)*inv(sF)*(x(i,:)-mF)'));
%     den1 = pie(1)*(norm(2*pi*sM)^(-1/2))*(exp(-1/2*(x(i,:)-mM)*inv(sM)*(x(i,:)-mM)'));
%     den2 = pie(2)*(norm(2*pi*sF)^(-1/2))*(exp(-1/2*(x(i,:)-mF)*inv(sF)*(x(i,:)-mF)'));
%     posteriorFF(i) = num/(den1+den2);
% end
% 
% x = testFemales;
% for i = 1:length(x)
%     num = pie(1)*(norm(2*pi*sM)^(-1/2))*(exp(-1/2*(x(i,:)-mM)*inv(sM)*(x(i,:)-mM)'));
%     den1 = pie(1)*(norm(2*pi*sM)^(-1/2))*(exp(-1/2*(x(i,:)-mM)*inv(sM)*(x(i,:)-mM)'));
%     den2 = pie(2)*(norm(2*pi*sF)^(-1/2))*(exp(-1/2*(x(i,:)-mF)*inv(sF)*(x(i,:)-mF)'));
%     posteriorFM(i) = num/(den1+den2);
%end

classified(run) = (sum(posteriorM(1:SPLIT_M) > posteriorF(1:SPLIT_M))+sum(posteriorM(SPLIT_M+1:end) < posteriorF(SPLIT_M+1:end))) /(SPLIT_M+SPLIT_F)

end

%% GRAPH

figure(5)
scatter(testMales(:,1),testMales(:,2),100, '.')
hold on
grid minor
scatter(testFemales(:,1),testFemales(:,2),100, '.')
M = (mean(testMales));
%scatter(M(1),M(2),100, 'kx')
F = (mean(testFemales));
%scatter(F(1),F(2),100, 'kx')
MF = mean([M; F]);
%scatter(MF(1),MF(2),100, 'mx')
axis equal
xlabel('Height')
ylabel('Weight')
legend('Males', 'Females', 'location','best')