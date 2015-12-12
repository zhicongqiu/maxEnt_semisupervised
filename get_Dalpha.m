function [Dalpha] = get_Dalpha(data,Fs,Fs_multiclassN,Fs_multiclassR,...
			       w,label_N,label_R,label,active_set_normal,...
			       active_set_rare,WN,WR,a_u,mode)
sum0 = 0;
sum1 = 0;
sum2 = 0;
sum2N = 0;
sum2R = 0;
for i=1:size(Fs,1)
    if label(i)==0 %known normal category
        sum0 = sum0+Fs(i)*data(i);
    elseif label(i)==1 %known rare category
        sum1 = sum1-(1-Fs(i))*data(i);
    %use unlabeled samples only if mode is 2 or 3 or 4
    elseif label(i)==2 && (mode==2||mode==3||mode==4)
        sum2 = sum2+(2*Fs(i)-1)*data(i); 
    end
end

%sum up the unlabeled samples
data_U = data(label==2);
Fs_U = Fs(label==2);

num_normal = sum(active_set_normal);
num_rare = sum(active_set_rare);
if num_normal>=2 && (mode==3||mode==2)
    Fn_U = Fs_multiclassN(label_N==2,:);
    tempN = find(active_set_normal==1);
    for i=1:size(Fs_U,1)
        for j=1:length(tempN)
            sum2N = sum2N + ...
                    Fs_U(i)*(1-Fs_U(i))*data_U(i)*...
		    (log(1/num_normal)-log(Fn_U(i,tempN(j))));
        end        
    end    
end
if num_rare>=2 && (mode==3||mode==2)
    Fr_U = Fs_multiclassR(label_R==2,:);
    tempR = find(active_set_rare==1);
    for i=1:size(Fs_U,1)    
        for j=1:length(tempR)               
            sum2R = sum2R -...
                Fs_U(i)*(1-Fs_U(i))*data_U(i)*...
		(log(1/num_rare)-log(Fr_U(i,tempR(j))));
        end
    end   
end

%make sure sum is used
if num_rare==0
    num_rare=1;
end
if WR==0
    if mode~=3&&mode~=1
        Dalpha = (1-a_u)*(sum0+sum1)...
            +a_u*(0.5*sum2+sum2N+sum2R);
    elseif mode==3
        Dalpha = (1-a_u)*(sum0+sum1)...
            -a_u*(0.5*sum2+sum2N+sum2R);   
    elseif mode==1
        Dalpha = (1-a_u)*(sum0+sum1)...
            +a_u*2*w;        
    end
else
    if mode~=3&&mode~=1
      Dalpha = (1-a_u)*(sum0+(WN/WR)*sum1)...
	       +a_u*(0.5*sum2+sum2N/num_normal+sum2R/num_rare);
    elseif mode==3
      Dalpha = (1-a_u)*(sum0+(WN/WR)*sum1)...
               -a_u*(0.5*sum2+sum2N/num_normal+sum2R/num_rare);
    elseif mode==1
      Dalpha = (1-a_u)*(sum0+(WN/WR)*sum1)...
               +a_u*2*w;        
    end
end
