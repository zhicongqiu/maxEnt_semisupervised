function [ROC_AUC ERROR SPARSITY] = semi_learning(data_raw,data_p,data_GT,test_raw,test_p,test_GT,normal_class,rare_class,num_train,mode,pval4R)
%input: raw data, p-value data, label coresponding to each data
%first num_train samples are used as labeled normal/rare samples
%0:MLE regularization or 1:L2 regularization or 2:proposed model or 3:
%min_ent
N_O = size(data_raw,2);
N_P = size(data_p,2);
M = length(data_GT);
label = 2*ones(M,1); %2 is unlabeled, 1 is rare, 0 is normal
%label(1:num_train) = 0;

%ground-truth transformation, 1st column indicates its Param index; 
%2nd column indicates if it is normal (0)
data_GTT = transform_GT(data_GT,normal_class,rare_class);
test_GTT = transform_GT(test_GT,normal_class,rare_class);

%initialize parameters
active_set_normal = zeros(length(normal_class),1);
active_set_rare = zeros(length(rare_class),1);
num_normal_labeled = 0;
num_rare_labeled = 0;
for i=1:num_train
    if data_GTT(i,2)==0
        if active_set_normal(data_GTT(i,1))==0
            active_set_normal(data_GTT(i,1)) = 1;
        end
        label(i) = 0;
        num_normal_labeled = num_normal_labeled+1;
        %{
    elseif data_GTT(i,2)==1
        if active_set_rare(data_GTT(i,1))==0
            active_set_rare(data_GTT(i,1)) = 1;
        end
        label(i) = 1;
        num_rare_labeled = num_rare_labeled+1;
    %}
    end
end
%label only one sample from each category
for i=num_train+1:M
    if data_GTT(i,2)==1
        if active_set_rare(data_GTT(i,1))==0
            active_set_rare(data_GTT(i,1)) = 1;
            label(i) = 1;
            num_rare_labeled = num_rare_labeled+1;
        end

    end
end


SPARSITY = struct;
ParamN = struct;
ParamR = struct;
ParamPR = struct;
for i=1:length(normal_class)    
    ParamN(i).beta = 1e-6*ones(1,N_O);
    ParamN(i).beta0 = 0;
end
for i=1:length(rare_class)   
    if pval4R==false
        ParamR(i).beta = 1e-6*ones(1,N_O);
    else
        ParamR(i).beta = 1e-6*ones(1,N_P);
    end
    ParamR(i).beta0 = 0;
end
ParamPR.alpha = 1e-6*ones(1,N_P);
ParamPR.alpha0 = 0;

    
    a_u = 0.2;%DoCV(data_raw,N_O,data_p,N_P,data_GTT,label,active_set_normal,active_set_rare,pval4R,true);
    disp(a_u);
    %0:MLE regularization or 1:L2 regularization or 2:proposed model or 3: min_ent 
    if pval4R==false
        [ParamPR ParamN ParamR] = ...
            gradDes(data_raw,data_p,data_GTT,label,a_u,ParamPR,ParamN,ParamR,active_set_normal,active_set_rare,num_normal_labeled,num_rare_labeled,pval4R,mode);
    else
        [ParamPR ParamN ParamR] = ...
            gradDes(data_p,data_p,data_GTT,label,a_u,ParamPR,ParamN,ParamR,active_set_normal,active_set_rare,num_normal_labeled,num_rare_labeled,pval4R,mode);	
    end
    %test PMF based on current parameters
    [test_PMFunknown test_PMFnormal test_PMFrare] = getTestPMF(test_raw,test_p,test_GTT,ParamPR,ParamN,ParamR,active_set_normal,active_set_rare,pval4R);
    
    %sparsity measure
    SPARSITY.ParamPR = ParamPR;
    SPARSITY.PR_mag = norm(ParamPR.alpha);
    SPARSITY.PR_sparse = sum(ParamPR.alpha==0)/length(ParamPR.alpha);    
    SPARSITY.ParamN = ParamN(active_set_normal==1);
    SPARSITY.ParamR = ParamR(active_set_rare==1);
    SPARSITY.active_set_normal = active_set_normal;
    SPARSITY.active_set_rare = active_set_rare;
    
    %test set performance measure
    ROC_AUC = calculate_ROC(test_PMFunknown,test_GTT);
    %display ROC_AUC
    disp(ROC_AUC);
    
    %error rate on test set
    [error FAR FNR error_CN error_CR error_avgC] = calculate_error(test_PMFunknown,test_PMFnormal,test_PMFrare,test_GTT,0);
    disp(error_avgC);
    
ERROR = struct;
ERROR.error = error; ERROR.FAR = FAR; ERROR.FNR = FNR;
ERROR.error_CN = error_CN; ERROR.error_CR = error_CR;
ERROR.avgC = error_avgC;

    
    