data = load('Indian_Pines_Dataset')
indian_pines = data.indian_pines;
indian_pines_gt = data.indian_pines_gt;
C1 = 237;  % Corn
C2 = 1265;  % Woods
N_SPECTR = 220;

n=0;
class1 = zeros(C1, N_SPECTR);
for i = 1:size(indian_pines, 1)
    for j = 1:size(indian_pines, 2)
        if indian_pines_gt(i,j)== 4 % class index
            n = n + 1;
            class1(n,:) = indian_pines(i,j,:);
        end
    end
end

n = 0
class2 = zeros(C2, N_SPECTR);
for i = 1:size(indian_pines, 1)
    for j = 1:size(indian_pines, 2)
        if indian_pines_gt(i,j)== 14 % class index
            n = n + 1;
            class2(n,:) = indian_pines(i,j,:);
        end
    end
end