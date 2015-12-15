function Dbeta = get_Dbeta(data,Fa,Fs_multiclass,w,data_GTT_temp,...
			   label_temp,l,active_set,num,WN,WR,a_u,mode,rare)

sum0 = 0;
sum1 = 0;
sum2 = 0;
for i=1:size(Fs_multiclass,1)
    if label_temp(i)~=2 %labeled class
        if data_GTT_temp(i,1)==l %same-class label
            sum0 = sum0-(1-Fs_multiclass(i,l))*data(i);
        else %diff-class label
            sum1 = sum1+Fs_multiclass(i,l)*data(i);
        end
    else %unknown class
        if mode==2||mode==3||mode==4
            temp = find(active_set==1);
            for j=1:length(temp)
                if temp(j)==l
                    sum2 = sum2-Fa(i)*(1-Fs_multiclass(i,l))*data(i);
                else
                    sum2 = sum2+Fa(i)*Fs_multiclass(i,l)*data(i);
                end
            end
        end
    end
end
if num==0
    num=1;
end

if rare==true
if mode~=3||mode~=1
    Dbeta = (1-a_u)*(WN/WR)*(sum0+sum1)+a_u*sum2/num;
elseif mode==3
    Dbeta = (1-a_u)*(WN/WR)*(sum0+sum1)-a_u*sum2/num;
elseif mode==1
    Dbeta = (1-a_u)*(WN/WR)*(sum0+sum1)+a_u*2*w;
end

else
if mode~=3||mode~=1
    Dbeta = (1-a_u)*(sum0+sum1)+a_u*sum2/num;
elseif mode==3
    Dbeta = (1-a_u)*(sum0+sum1)-a_u*sum2/num;
elseif mode==1
    Dbeta = (1-a_u)*(sum0+sum1)+a_u*2*w;
end
end
