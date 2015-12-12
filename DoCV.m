function [a_u] = DoCV(data_raw,N_O,data_p,N_P,data_GTT,...
		      label,active_set_normal,active_set_rare,...
		      reinitialize,pos,mode,rare_exist,weighted)

start = 0.9;
ParamN = struct;
ParamR = struct;
ParamPR = struct;
%10-fold CV
nr_fold = 10;

if sum(active_set_rare)==0 %no suspicious sample, choose a_u=0.1
    a_u = 0.2;
    return;
else
    %we have label_train_set,label_test_set and unlabeled samples
    train_raw = data_raw(label~=2,:);
    train_p = data_p(label~=2,:);
    train_GTT = data_GTT(label~=2,:);
    train_label = label(label~=2);
    train_M = size(train_raw,1);
    
    unlabeled_raw = data_raw(label==2,:);
    unlabeled_p = data_p(label==2,:);
    unlabeled_GTT = data_GTT(label==2,:);
    unlabeled_label = label(label==2);
    
    %create 10-fold
    rand_index = randsample(train_M,train_M);
    fold_start(1)=0; %for numerical reason
    for i=1:nr_fold
        temp = round(i*train_M/nr_fold);
        fold_start(i+1) = temp;
    end

    temp_error = 1;
    for i=start:-0.1:0.1
        %reinitialize parameter?
        if reinitialize==true||i==start
            for k=1:length(active_set_normal)    
              ParamN(k).beta = 1e-6*ones(1,N_O);
              ParamN(k).beta0 = 0;
            end
	    if rare_exist
              for k=1:length(active_set_rare)  
		ParamR(k).beta = 1e-6*ones(1,N_O);
		ParamR(k).beta0 = 0;
              end
              ParamPR.alpha = 1e-6*ones(1,N_P);
              ParamPR.alpha0 = 0;
	    end
        end
        
        total_error_normal = 0;
        total_error_rare = 0;
	total_errorCN = 0;
        %10-fold CV
        for j=1:nr_fold
          %1-fold for testing
          testCV_raw = ...
	  train_raw(rand_index(fold_start(j)+1:fold_start(j+1)),:);
          testCV_p = ...
	  train_p(rand_index(fold_start(j)+1:fold_start(j+1)),:);
          testCV_GTT = ...
	  train_GTT(rand_index(fold_start(j)+1:fold_start(j+1)),:);
          testCV_label = ...
	  train_label(rand_index(fold_start(j)+1:fold_start(j+1)),:);
            
          %the other folds for training
          trainCV_raw = ...
          [train_raw([rand_index(1:fold_start(j));rand_index(fold_start(j+1)+1:end)],:);unlabeled_raw];
          trainCV_p = ...
          [train_p([rand_index(1:fold_start(j));rand_index(fold_start(j+1)+1:end)],:);unlabeled_p];
          trainCV_GTT = ...
          [train_GTT([rand_index(1:fold_start(j));rand_index(fold_start(j+1)+1:end)],:);unlabeled_GTT];
          trainCV_label = ...
          [train_label([rand_index(1:fold_start(j));rand_index(fold_start(j+1)+1:end)],:);unlabeled_label];
	  if weighted
            trainCV_numN = sum(trainCV_label==0);
            trainCV_numR = sum(trainCV_label==1);
	  else
            trainCV_numN = 1;
            trainCV_numR = 1;
	  end
          %active set redefined
          active_set_normalT = zeros(length(active_set_normal),1);
          if sum(active_set_normal)>=2               
            for k=1:size(trainCV_p,1)
              if trainCV_label(k)==0&&active_set_normalT(trainCV_GTT(k,1))==0
                active_set_normalT(trainCV_GTT(k,1)) = 1;
              end
            end
          end
	  active_set_rareT = zeros(length(active_set_rare),1);
	  if rare_exist
            %rare set redefined    
            if sum(active_set_rare)>=2               
              for k=1:size(trainCV_p,1)
		if trainCV_label(k)==1&&active_set_rareT(trainCV_GTT(k,1))==0
                  active_set_rareT(trainCV_GTT(k,1)) = 1;
		end
              end
	    end
	  end

          %gradient descend for training fold
          [ParamPR ParamN ParamR] = ...
          gradDes(trainCV_raw,trainCV_p,trainCV_GTT,trainCV_label,...
		  i,ParamPR,ParamN,ParamR,active_set_normalT,...
		  active_set_rareT,trainCV_numN,trainCV_numR,...
		  pos,mode,rare_exist);
          %use classification error, unweighted or weighted???
          [test_PMFunknown test_PMFnormal test_PMFrare] = ...
	  getTestPMF(testCV_raw,testCV_p,testCV_GTT,...
		     ParamPR,ParamN,ParamR,...
		     active_set_normalT,active_set_rareT,rare_exist);  
          [error FA FN error_CN error_CR error_avgC] = ...
	  calculate_error(test_PMFunknown,test_PMFnormal,...
			  test_PMFrare,testCV_GTT,1,rare_exist);
          %account for both inter-class and intra-class errors?
          %total_error_normal = total_error_normal+FA+error_CN;
          %total_error_rare = total_error_rare+FN+error_CR;
          if rare_exist
            total_error_normal = total_error_normal+FA;
            total_error_rare = total_error_rare+FN;            
	  else
	    total_errorCN = total_errorCN+error_avgC;
	  end
        end
	temp_sum = sum(label==0);
	if rare_exist
	  %if all labeled samples are normal, use standard error
          if temp_sum==size(train_raw,1) 
            weighted_error = total_error_normal/temp_sum;
          else
            weighted_error = ...
            ((size(train_raw,1)-temp_sum)/size(train_raw,1))*...
	    total_error_normal/temp_sum +...
            (temp_sum/size(train_raw,1))*...
	    total_error_rare/(size(train_raw,1)-temp_sum);
          end
	else
	    weighted_error = total_errorCN/nr_fold;
	end
        %disp(weighted_error);       
        if weighted_error <= temp_error
          temp_error = weighted_error;
          a_u = i;            
        end
        
    end
    
end
