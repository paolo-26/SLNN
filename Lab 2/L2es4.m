clear; close all; clc

%% DATA
data = load('heightWeight')
SPLIT_M = 25
SPLIT_F = 40

males = data.heightWeightData(data.heightWeightData(:,1) == 1,2:end);
females = data.heightWeightData(data.heightWeightData(:,1) == 2,2:end);

test_males = males(1:SPLIT_M,:);
train_males = males(SPLIT_M+1:end,:);

test_females = females(1:SPLIT_F,:);
train_females = females(SPLIT_F+1:end,:);

for run = 1:3
%% PART 1
% MLE mean (males).
mM = 0;
for i = 1:length(train_males)
    mM = mM + train_males(i,:);
end
mM = mM/length(train_males)

% MLE mean (females).
mF = 0;
for i = 1:length(train_females)
    mF = mF + train_females(i,:);
end
mF = mF/length(train_females)

% MLE covariance (males).
firstTerm = zeros(2);  % 2x2 matrix of zeros
for i = 1:length(train_males)
    firstTerm = firstTerm + train_males(i,:)'*train_males(i,:); 
end
firstTerm = firstTerm/length(train_males);
secondTerm = mM.*mM';
sM = firstTerm - secondTerm

if run == 2
   sM = diag(diag(sM)) 
end

if run == 3
    firstTerm = zeros(2);  % 2x2 matrix of zeros
    shared = [train_males; train_females]
    for i = 1:length(shared)
        firstTerm = firstTerm + shared(i,:)'*shared(i,:); 
    end
    firstTerm = firstTerm/length(shared);
    secondTerm = mM.*mM';
    sM = firstTerm - secondTerm
end

% MLE covariance (females).
firstTerm = zeros(2);  % 2x2 matrix of zeros
for i = 1:length(train_females)
    firstTerm = firstTerm + train_females(i,:)'*train_females(i,:); 
end
firstTerm = firstTerm/length(train_females);
secondTerm = mF.*mF';
sF = firstTerm - secondTerm

if run == 2
   sF = diag(diag(sF)) 
end

if run == 3
   sf = sM 
end
L = length(data.heightWeightData);
pie(1) = [(L-SPLIT_M)/((L-SPLIT_M)+(L-SPLIT_F))];
pie(2) = [(L-SPLIT_F)/((L-SPLIT_M)+(L-SPLIT_F))];

%% PART 1

x = [test_males; test_females];
for i = 1:length(x)
    num1 = pie(1)*(norm(2*pi*sM)^(-1/2))*(exp(-1/2*(x(i,:)-mM)*inv(sM)*(x(i,:)-mM)'));
    den1 = pie(1)*(norm(2*pi*sM)^(-1/2))*(exp(-1/2*(x(i,:)-mM)*inv(sM)*(x(i,:)-mM)'));
    den2 = pie(2)*(norm(2*pi*sF)^(-1/2))*(exp(-1/2*(x(i,:)-mF)*inv(sF)*(x(i,:)-mF)'));
    posteriorM(i) = num1/(den1+den2);
end

x = [test_males; test_females];
for i = 1:length(x)
    num1 = pie(1)*(norm(2*pi*sF)^(-1/2))*(exp(-1/2*(x(i,:)-mF)*inv(sF)*(x(i,:)-mF)'));
    den1 = pie(1)*(norm(2*pi*sM)^(-1/2))*(exp(-1/2*(x(i,:)-mM)*inv(sM)*(x(i,:)-mM)'));
    den2 = pie(2)*(norm(2*pi*sF)^(-1/2))*(exp(-1/2*(x(i,:)-mF)*inv(sF)*(x(i,:)-mF)'));
    posteriorF(i) = num1/(den1+den2);
end
% 
% x = test_females;
% for i = 1:length(x)
%     num1 = pie(1)*(norm(2*pi*sF)^(-1/2))*(exp(-1/2*(x(i,:)-mF)*inv(sF)*(x(i,:)-mF)'));
%     den1 = pie(1)*(norm(2*pi*sM)^(-1/2))*(exp(-1/2*(x(i,:)-mM)*inv(sM)*(x(i,:)-mM)'));
%     den2 = pie(2)*(norm(2*pi*sF)^(-1/2))*(exp(-1/2*(x(i,:)-mF)*inv(sF)*(x(i,:)-mF)'));
%     posteriorFF(i) = num1/(den1+den2);
% end
% 
% x = test_females;
% for i = 1:length(x)
%     num1 = pie(1)*(norm(2*pi*sM)^(-1/2))*(exp(-1/2*(x(i,:)-mM)*inv(sM)*(x(i,:)-mM)'));
%     den1 = pie(1)*(norm(2*pi*sM)^(-1/2))*(exp(-1/2*(x(i,:)-mM)*inv(sM)*(x(i,:)-mM)'));
%     den2 = pie(2)*(norm(2*pi*sF)^(-1/2))*(exp(-1/2*(x(i,:)-mF)*inv(sF)*(x(i,:)-mF)'));
%     posteriorFM(i) = num1/(den1+den2);
%end

classified(run) = (sum(posteriorM(1:SPLIT_M) > posteriorF(1:SPLIT_M))+sum(posteriorM(SPLIT_M+1:end) < posteriorF(SPLIT_M+1:end))) /(SPLIT_M+SPLIT_F)

end

%% GRAPH

figure(5)
scatter(test_males(:,1),test_males(:,2), '.')
hold on
grid minor
scatter(test_females(:,1),test_females(:,2), '.')
M = (mean(test_males));
%scatter(M(1),M(2),100, 'kx')
F = (mean(test_females));
%scatter(F(1),F(2),100, 'kx')
MF = mean([M; F]);
scatter(MF(1),MF(2),100, 'mx')
axis equal
xlabel('Height')
ylabel('Weight')
legend('Males', 'Females', 'location','best')