function [ROC_AUC ERROR SPARSITY] = semi_learning(data_raw,data_p,data_GT,...
						  test_raw,test_p,test_GT,...
						  normal_class,rare_class,...
						  num_train,mode,rare_exist,pos,weighted)
%[ERROR test_PMFunknown ENT ParamPR] = ...
%semi_learning_toy(data_raw,data_GT,test_raw,test_GT,normal_class,rare_class,num_train,pos,mode)
%input: training data in either raw or p-val space, labels, test data, test labels
%first num_train samples in training data from both classes are used as labeled normal/rare samples
%mode:
%0:MLE regularization 
%1:L2 regularization 
%2: proposed maxEnt regularizer 
%3: min_ent regularizer
%rare_exist: 
%1: enable rare category classification
%0: standard LR classification on multiple classes
%pos: 
%1/true to impose positive constraint
%weighted:
%1/true to use weighting
%output:
%ERROR metrics, test pmf, test sample entropy and parameters

%check input consistency
if ~ismember(rare_exist,[0 1])
   error('please specify if rare classes exist, 0 if not used')
end
%make sure there is the specified mode
if ~ismember(mode,[0 1 2 3])
   error('please specify modes: 0, 1, 2 or 3')
end
%make sure train and test has the same input dimension
if size(data_raw,2)~=size(test_raw,2) || size(data_p,2)~=size(test_p,2)
   error('train and test dimension mismatch')
end
%make sure number of samples is the same as number of labels
if length(data_raw)~=length(data_GT)||length(data_p)~=length(data_GT)
   error('number of training samples should be equal to number of labels')
end
if length(test_raw)~=length(data_GT)||length(test_p)~=length(data_GT)
   error('number of test samples should be equal to number of labels')
end


[N K_O] = size(data_raw);
if num_train>N
   num_train = N
end

K_P = size(data_p,2);
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
WN = 0;
WR = 0;
for i=1:num_train
    if data_GTT(i,2)==0
        if active_set_normal(data_GTT(i,1))==0
            active_set_normal(data_GTT(i,1)) = 1;
        end
        label(i) = 0;
        WN = WN+1;
        
    elseif data_GTT(i,2)==1
        if active_set_rare(data_GTT(i,1))==0
            active_set_rare(data_GTT(i,1)) = 1;
        end
        label(i) = 1;
        WR = WR+1;
    end
end
%{
%label only one sample from each category
for i=num_train+1:M
    if data_GTT(i,2)==1
        if active_set_rare(data_GTT(i,1))==0
            active_set_rare(data_GTT(i,1)) = 1;
            label(i) = 1;
            WR = WR+1;
        end
    end
end
%}

SPARSITY = struct;
ParamN = struct;
ParamR = struct;
ParamPR = struct;
for i=1:length(normal_class)    
  ParamN(i).beta = 1e-6*ones(1,K_O);
  ParamN(i).beta0 = 0;
end
for i=1:length(rare_class) 
  %{
    if pval4R==false
        ParamR(i).beta = 1e-6*ones(1,K_O);
    else
        ParamR(i).beta = 1e-6*ones(1,K_P);
    end
  %}
  ParamR(i).beta = 1e-6*ones(1,K_O);
  ParamR(i).beta0 = 0;
end
ParamPR.alpha = 1e-6*ones(1,K_P);
ParamPR.alpha0 = 0;

    
a_u = 0.2;%DoCV(data_raw,N_O,data_p,N_P,data_GTT,label,active_set_normal,active_set_rare,pval4R,true);
disp(a_u);
if weighted
  WN = 1;WR = 1;
end
%0:MLE regularization or 1:L2 regularization or 2:proposed model or 3: min_ent 
%if pval4R==false
[ParamPR ParamN ParamR] = ...
gradDes(data_raw,data_p,data_GTT,label,a_u,ParamPR,ParamN,ParamR,...
	active_set_normal,active_set_rare,...
	WN,WR,pos,mode);
%{
else
  [ParamPR ParamN ParamR] = ...
  gradDes(data_p,data_p,data_GTT,label,a_u,ParamPR,ParamN,ParamR,...
	  active_set_normal,active_set_rare,WN,WR,pval4R,mode);	
end
%}
%test PMF based on current parameters
[test_PMFunknown test_PMFnormal test_PMFrare] = ...
getTestPMF(test_raw,test_p,test_GTT,ParamPR,ParamN,ParamR,active_set_normal,active_set_rare);
    
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
[error FAR FNR error_CN error_CR error_avgC] = ...
calculate_error(test_PMFunknown,test_PMFnormal,test_PMFrare,test_GTT,0);
disp(error_avgC);
    
ERROR = struct;
ERROR.error = error; ERROR.FAR = FAR; ERROR.FNR = FNR;
ERROR.error_CN = error_CN; ERROR.error_CR = error_CR;
ERROR.avgC = error_avgC;

    
    
