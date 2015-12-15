function [ParamPR_old ParamN_old ParamR_old] = ...
         gradDes(data_raw,data_p,data_GTT,label,a_u,...
		 ParamPR_old,ParamN_old,ParamR_old,...
		 active_set_normal,active_set_rare,...
		 WN,WR,pos,mode,rare_exist)

%initial step size and tolerance level
step_size = 1e-3;
reltol = 1e-6;

%initialize inter-normal and inter-rare
data_GTT_tempR = 0;
data_tempR = 0;
label_tempR = 0;
ParamR_new = ParamR_old;
ParamPR_new = ParamPR_old;
if rare_exist
    L2 = norm([ParamPR_old.alpha0 ParamPR_old.alpha])^2;
    %top level uses p-value
    Fs = getF(-data_p,ParamPR_old);
    num_active_rare = sum(active_set_rare);
    if num_active_rare>=2
        tempR = find(active_set_rare==1);
        %first,filter out rare samples
        data_tempR = data_raw(label~=0,:);
        data_GTT_tempR = data_GTT(label~=0,:);
        label_tempR = label(label~=0);
        %ParamR_new = ParamR_old;
        %ParamPR_new = ParamPR_old;
        Fs_multiclassR = ...
        getF_multiclass(data_tempR,data_GTT_tempR,label_tempR,...
                ParamR_old,active_set_rare);
        for i=1:length(tempR)-1
            L2 = L2 + ...
            norm([ParamR_old(tempR(i)).beta0 ParamR_old(tempR(i)).beta])^2;
        end
    else
        Fs_multiclassR = [];
    end
else
    L2 = 0;
    num_active_rare = 0;
    Fs = [];
    Fs_multiclassR = [];
end
num_active_normal = sum(active_set_normal);
data_GTT_tempN = 0;
data_tempN = 0;
label_tempN = 0;
ParamN_new = ParamN_old;
if num_active_normal>=2
    tempN = find(active_set_normal==1);
    %first,filter out rare samples
    data_tempN = data_raw(label~=1,:);
    data_GTT_tempN = data_GTT(label~=1,:);
    label_tempN = label(label~=1);
end
Fs_multiclassN = 0;
if num_active_normal>=2
    Fs_multiclassN = ...
    getF_multiclass(data_tempN,data_GTT_tempN,label_tempN,...
		    ParamN_old,active_set_normal);
    for i=1:length(tempN)-1
        L2 = L2 + ...
	     norm([ParamN_old(tempN(i)).beta0 ParamN_old(tempN(i)).beta])^2;
    end
end

%calculate initial objective function value
Dold =  ObjF_meta(Fs,Fs_multiclassN,Fs_multiclassR,L2,...
		  data_GTT_tempN,data_GTT_tempR,label_tempN,label_tempR,...
		  label,active_set_normal,active_set_rare,...
		  WN,WR,a_u,mode,rare_exist);
%disp(Dold);
Dnew = inf;
updated = true;
mu = step_size;
count_step = 0;
while abs(Dnew-Dold)>=reltol||Dnew>Dold||isinf(Dold) 
    
    %if num_active_rare>=2
        %disp(Dold);
    %end
    count_step = count_step+1;
    fprintf('iteration %d; diff is %g\n',count_step,abs(Dnew-Dold));
    if updated==true
       temp_normal = 0;    
       if num_active_normal>=2
         Fs_multiclassN = ...
         getF_multiclass(data_tempN,data_GTT_tempN,label_tempN,...
         ParamN_old,active_set_normal);
       end

       if rare_exist
        temp_PR = 0;
        temp_rare = 0;
        Fs = getF(-data_p,ParamPR_old);
         if num_active_rare>=2
           Fs_multiclassR = ...
	   getF_multiclass(data_tempR,data_GTT_tempR,label_tempR,...
			   ParamR_old,active_set_rare);
         end 

         Dalpha0 = ...
	 get_Dalpha(ones(size(data_raw,1),1),Fs,...
		    Fs_multiclassN,Fs_multiclassR,...
		    ParamPR_old.alpha0,label_tempN,label_tempR,label,...
		    active_set_normal,active_set_rare,WN,WR,a_u,mode);        
         for i=1:size(data_raw,2)
           Dalpha(i) = ... 
	   get_Dalpha(-data_p(:,i),Fs,Fs_multiclassN,Fs_multiclassR,...
		      ParamPR_old.alpha(i),label_tempN,label_tempR,label,...
		      active_set_normal,active_set_rare,WN,WR,a_u,mode);
         end
         temp_PR = temp_PR+Dalpha0^2+norm(Dalpha)^2;
       else
           Fs = ones(size(data_raw,1),1);
       end
    
       if num_active_normal>=2
         %update parameters for each class           
         %initialize Dbeta0 and Dbeta
         Dbeta0N = zeros(length(tempN)-1,1);
         DbetaN = zeros(length(tempN)-1,size(data_raw,2));           
         for i=1:length(tempN)-1
            Dbeta0N(i) = ...
            get_Dbeta(ones(size(data_tempN,1),1),Fs(label~=1),Fs_multiclassN,...
		     ParamN_old(tempN(i)).beta0,data_GTT_tempN,...
		     label_tempN,tempN(i),active_set_normal,...
		     num_active_normal,WN,WR,a_u,mode,false);
           temp_normal = temp_normal+Dbeta0N(i)^2;
           for j=1:size(data_raw,2)
             DbetaN(i,j) = ...
             get_Dbeta(data_tempN(:,j),Fs(label~=1),Fs_multiclassN,...
		       ParamN_old(tempN(i)).beta(j),data_GTT_tempN,...
		       label_tempN,tempN(i),active_set_normal,...
		       num_active_normal,WN,WR,a_u,mode,false);
           end
           temp_normal = temp_normal+norm(DbetaN(i,:))^2;
         end
       end            
       if num_active_rare>=2
         %update parameters for each class           
         %initialize Dbeta0 and Dbeta
         Dbeta0R = zeros(length(tempR)-1,1);
         DbetaR = zeros(length(tempR)-1,size(data_raw,2));           
         for i=1:length(tempR)-1
           Dbeta0R(i) = ...
           get_Dbeta(ones(size(data_tempR,1),1),Fs(label~=0),Fs_multiclassR,...
		     ParamR_old(tempR(i)).beta0,data_GTT_tempR,...
		     label_tempR,tempR(i),active_set_rare,...
		     num_active_rare,WN,WR,a_u,mode,true,rare_exist);
           temp_rare = temp_rare+Dbeta0R(i)^2;
           for j=1:size(data_raw,2)
             DbetaR(i,j) = ...
             get_Dbeta(-1*data_tempR(:,j),Fs(label~=0),Fs_multiclassR,...
		       ParamR_old(tempR(i)).beta(j),data_GTT_tempR,...
		       label_tempR,tempR(i),active_set_rare,...
		       num_active_rare,WN,WR,a_u,mode,true,rare_exist);
           end
           temp_rare = temp_rare+norm(DbetaR(i,:))^2;
         end
       end                    
       %temp_normALL = norm([temp_PR temp_normal temp_rare]); 
       if count_step>1
         Dold = Dnew;
       end
    end

    if rare_exist
      ParamPR_new.alpha0 = ParamPR_old.alpha0-mu*Dalpha0;%/norm(temp_PR);
      ParamPR_new.alpha = ParamPR_old.alpha-mu.*Dalpha;%./norm(temp_PR);   
      %positive constraints, projected gradient
      if pos
        ParamPR_new.alpha(ParamPR_new.alpha<0) = 0;
      end
    
      Fs = getF(-data_p,ParamPR_new);  
      L2 = norm([ParamPR_new.alpha0 ParamPR_new.alpha])^2;

      %inter-rare update
      if num_active_rare>=2
        for i=1:length(tempR)-1    
            ParamR_new(tempR(i)).beta0 = ...
	    ParamR_old(tempR(i)).beta0-mu*Dbeta0R(i);%/norm(temp_rare);
            %unconstraint weights
            ParamR_new(tempR(i)).beta = ...
	    ParamR_old(tempR(i)).beta-mu*DbetaR(i,:);%./norm(temp_rare);
        end
        Fs_multiclassR = ...
        getF_multiclass(data_tempR,data_GTT_tempR,label_tempR,...
        ParamR_new,active_set_rare);
        for i=1:length(tempR)-1
            L2 = L2 + ...
		 norm([ParamR_new(tempR(i)).beta0 ParamR_new(tempR(i)).beta])^2;
        end
      end 
    else
        L2 = 0;
    end

    %inter-normal update
    if num_active_normal>=2
        for i=1:length(tempN)-1
            ParamN_new(tempN(i)).beta0 = ...
            ParamN_old(tempN(i)).beta0-mu*Dbeta0N(i);%/norm(temp_normal);
            %unconstraint weights
            ParamN_new(tempN(i)).beta = ...
            ParamN_old(tempN(i)).beta-mu*DbetaN(i,:);%./norm(temp_normal);
        end
        Fs_multiclassN = ...
        getF_multiclass(data_tempN,data_GTT_tempN,label_tempN,...
        ParamN_new,active_set_normal);
        for i=1:length(tempN)-1
            L2 = L2 + ...
            norm([ParamN_new(tempN(i)).beta0 ParamN_new(tempN(i)).beta])^2;
        end
    end
   
    Dnew = ...
    ObjF_meta(Fs,Fs_multiclassN,Fs_multiclassR,L2,...
    data_GTT_tempN,data_GTT_tempR,...
    label_tempN,label_tempR,label,...
    active_set_normal,active_set_rare,WN,WR,a_u,mode,rare_exist);
    %fprintf('new is %f, old is %f\n',Dnew,Dold);
    if Dnew<Dold
        updated = true;
        mu = step_size;
	if rare_exist
          ParamPR_old = ParamPR_new;
          if num_active_rare>=2
            ParamR_old = ParamR_new;
          end
	end
        if num_active_normal>=2
            ParamN_old = ParamN_new;
        end

    else
        updated = false;
        %reduce step size
        mu = 0.5*mu;
    end
        
end
%disp(count_step);
%disp(Dold);
end
    
    
