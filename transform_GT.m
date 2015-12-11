function data_GTT = transform_GT(GT,normal_class,rare_class)

data_GTT = zeros(length(GT),2);
for i=1:length(GT)
    temp = find(GT(i)==normal_class);
    if isempty(temp)
        temp = find(GT(i)==rare_class);
        if isempty(temp)
            disp('a sample has no ground-truth?');
        end
        data_GTT(i,1)=temp;
        data_GTT(i,2) = 1;
    else
        data_GTT(i,1)=temp;
        data_GTT(i,2) = 0;
    end
end