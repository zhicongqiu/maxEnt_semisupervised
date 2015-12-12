function Fs_multiclass = getF_multiclass(data,GTT,label,Param,active_set)

N = size(data,1);

for i=length(active_set):-1:1
    if active_set(i)==1
        last_label = i;
        break;
    end
end

temp = find(active_set==1);
Fs_multiclass = zeros(N,length(active_set));
for i=1:N
    sum = 1;
    for j=1:length(temp)-1 %use the first k-1 classes
        sum = sum+exp(Param(temp(j)).beta0+data(i,:)*Param(temp(j)).beta');
    end
      
        for j=1:length(temp)-1 %use the first k-1 classes
            Fs_multiclass(i,temp(j)) = ...
                exp(Param(temp(j)).beta0+data(i,:)*Param(temp(j)).beta')/sum;
            if Fs_multiclass(i,temp(j))>1-1e-10
                Fs_multiclass(i,temp(j))=1-1e-10;
            elseif Fs_multiclass(i,temp(j))<1e-10
                Fs_multiclass(i,temp(j))=1e-10;
            end
        end
        Fs_multiclass(i,temp(end)) = 1/sum;
        if Fs_multiclass(i,temp(end))>1-1e-10
            Fs_multiclass(i,temp(end))=1-1e-10;
        elseif Fs_multiclass(i,temp(end))<1e-10
            Fs_multiclass(i,temp(end))=1e-10;
        end        
end
            
        
    
