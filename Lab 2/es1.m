close all; clc; clear
data=load ('heightWeight.mat')

cartadazucchero = [137; 207; 240]/255;
ametista = [153; 102; 204]/255;

male=data.heightWeightData(data.heightWeightData(:,1)==1,2:end);
female=data.heightWeightData(data.heightWeightData(:,1)==2,2:end);

figure(1)
hold on
scatter(male(:,1),male(:,2),'xb')
scatter(female(:,1),female(:,2),'pm')
grid on
grid minor
xlabel('Height (cm)')
ylabel('Weight (kg)')
legend('Male','Female','location','northwest')

figure(2)
hold on
edges=130:5:210;
h1 = histcounts(male(:,1),edges);
h2 = histcounts(female(:,1),edges);
b = bar(edges(1:end-1),[h1;h2]',1);
b(1).FaceColor = cartadazucchero;
b(2).FaceColor = ametista;
xlabel('Height (cm)')
grid on
grid minor
 legend('Male','Female','location','northeast')
 
figure(3)
hold on
edges=50:5:130;
h1 = histcounts(male(:,2),edges);
h2 = histcounts(female(:,2),edges);
b = bar(edges(1:end-1),[h1;h2]',1);
b(1).FaceColor = cartadazucchero;
b(2).FaceColor = ametista;
xlabel('Weight (kg)')
grid on
grid minor
legend('Male','Female','location','northeast')

%figure(2)
% hold on
% histogram(male(:,1),'binwidth',1)
% histogram(female(:,1),'binwidth',1)
% grid on
% grid minor
% xlabel('Height (cm)')
% ylabel('Weight (kg)')
% legend('Male','Female','location','northwest')
% 
% figure(3)
% hold on
% histogram(male(:,2),'binwidth',1)
% histogram(female(:,2),'binwidth',1)
% grid on
% grid minor
% xlabel('Height (cm)')
% ylabel('Weight (kg)')
% legend('Male','Female','location','northwest')

m_male = [mean(male)]
m_female = [mean(female)]
c_male = [cov(male)]
c_female = [cov(female)]

figure(4)
x1 = 120:220; x2 = 30:130;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],m_male,c_male);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
colormap parula
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([120 220 30 130 0 max(F(:))])
xlabel('Height (cm)'); ylabel('Weight (kg)'); zlabel('Probability Density - males');
title('Males')
colormap parula; view(0,90); axis equal; colorbar;

figure(5)
x1 = 120:220; x2 = 30:130;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],m_female,c_female);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([120 220 30 130 0 max(F(:))])
xlabel('Height (cm)'); ylabel('Weight (kg)'); zlabel('Probability Density - females');
title('Females')
colormap parula; view(0,90); axis equal; colorbar;