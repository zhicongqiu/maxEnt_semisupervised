function ROC_AUC = calculate_ROC(test_PMFunknown,test_GTT)

temp = sortrows([test_PMFunknown test_GTT(:,2)],-1);

FA = 0;
TD = 0;
area = 0;
for i=1:size(temp,1)
    if temp(i,2)==1 %rare sample
        TD = TD+1;   
    else
        area = area+TD;
        FA = FA+1;
    end
end
ROC_AUC = area/(TD*FA);