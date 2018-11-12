close all; clc; clear
data = load('heightWeight.mat')

cartaDaZucchero = [137; 207; 240]/255;  % Color for plots
ametista = [153; 102; 204]/255;  % Color for plots

male = data.heightWeightData(data.heightWeightData(:,1) == 1,2:end);
female = data.heightWeightData(data.heightWeightData(:,1) == 2,2:end);

figure(1)  % Scatter plot (males and females)
hold on
scatter(male(:,1),male(:,2),'^b')
scatter(female(:,1),female(:,2),'vm')
grid on
grid minor
xlabel('Height (cm)')
ylabel('Weight (kg)')
legend('Males','Females','location','northwest')
xlim([120 220])
ylim([30 130])
axis equal

figure(2)  % Histogram (males)
hold on
edges=120:5:220;
h1 = histcounts(male(:,1),edges);
h2 = histcounts(female(:,1),edges);
b = bar(edges(1:end-1),[h1;h2]',1);
b(1).FaceColor = cartaDaZucchero;
b(2).FaceColor = ametista;
xlabel('Height (cm)')
ylabel('Number of people')
grid on
grid minor
 legend('Males','Females','location','northeast')
 
figure(3)  % Histogram (females)
hold on
edges=30:5:130;
h1 = histcounts(male(:,2),edges);
h2 = histcounts(female(:,2),edges);
b = bar(edges(1:end-1),[h1;h2]',1);
b(1).FaceColor = cartaDaZucchero;
b(2).FaceColor = ametista;
xlabel('Weight (kg)')
ylabel('Number of people')
grid on
grid minor
legend('Male','Female','location','northeast')

% MLE mean (males).
mMales = 0;
for i = 1:length(male)
    mMales = mMales + male(i,:);
end
mMales = mMales/length(male)

% MLE mean (females).
mFemales = 0;
for i = 1:length(female)
    mFemales = mFemales + female(i,:);
end
mFemales = mFemales/length(female)

% MLE covariance (males).
firstTerm = zeros(2);  % 2x2 matrix of zeros
for i = 1:length(male)
    firstTerm = firstTerm + male(i,:)'*male(i,:); 
end
firstTerm = firstTerm/length(male);
secondTerm = mMales.*mMales';
sigmaMales = firstTerm - secondTerm

% MLE covariance (females).
firstTerm = zeros(2);  % 2x2 matrix of zeros
for i = 1:length(female)
    firstTerm = firstTerm + female(i,:)'*female(i,:); 
end
firstTerm = firstTerm/length(female);
secondTerm = mFemales.*mFemales';
sigmaFemales = firstTerm - secondTerm

% Multivariate gaussian plots.
figure(4)
x1 = 120:220; x2 = 30:130;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mMales,sigmaMales);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([120 220 30 130 0 max(F(:))])
xlabel('Height (cm)'); ylabel('Weight (kg)'); zlabel('Probability Density - males');
title('Males')
colormap parula; view(0,90); axis equal; colorbar;

figure(5)
x1 = 120:220; x2 = 30:130;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mFemales,sigmaFemales);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([120 220 30 130 0 max(F(:))])
xlabel('Height (cm)'); ylabel('Weight (kg)'); zlabel('Probability Density - females');
title('Females')
colormap parula; view(0,90); axis equal; colorbar;