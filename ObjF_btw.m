function  D = ObjF_btw(data,Fs,Param,numN,numR,a_u,data_GTT,label,active_set,rare)

Fs_multiclass = ...
    getF_multiclass(data,data_GTT,label,Param,active_set);
sum0 = 0;
sum2 = 0;
%assume Param has k classes, data_raw has no rare categories
temp = find(active_set==1);
for i=1:size(data,1)
    if label(i)~=2 %normal sample
        sum0 = sum0 - log(Fs_multiclass(i,data_GTT(i,1)));
    else  %unlabeled samples            
            for j=1:length(temp)
                sum2 = sum2+Fs(i)*log(Fs_multiclass(i,temp(j)));
            end
    end
end


D = (1-a_u)*sum0 - a_u/length(temp)*sum2;
