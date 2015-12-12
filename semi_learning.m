function [ROC_AUC ERROR SPARSITY] = ...
	 semi_learning(data_raw,data_p,data_GT,...
		       test_raw,test_p,test_GT,...
		       normal_class,rare_class,...
		       num_train,mode,rare_exist,pos,weighted)
%function [ROC_AUC ERROR SPARSITY] = ...
%	 semi_learning(data_raw,data_p,data_GT,...
%		       test_raw,test_p,test_GT,...
%		       normal_class,rare_class,...
%		       num_train,mode,rare_exist,pos,weighted)
%input: 
%training data, N samples and K features in either raw or p-val space, labels, 
%test data in either raw or p-val space, labels
%first num_train samples are used as labeled normal/rare samples
%mode:
%0:MLE regularization 
%1:L2 regularization 
%2: proposed maxEnt regularizer 
%3: min_ent regularizer
%rare_exist: 
%1: enable rare category classification
%0: standard LR classification on multiple classes
%pos: 
%1/true to impose positive constraint on p-val space
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
label = 2*ones(N,1); %2 is unlabeled, 1 is rare, 0 is normal
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

if sum(active_set_normal)<2 && rare_exist==false
   fprintf('there is one class labeled in standard LR learning...')
end

SPARSITY = struct;
ERROR = struct;
ParamN = struct;
ParamR = struct;
ParamPR = struct;
for i=1:length(normal_class)    
  ParamN(i).beta = 1e-6*ones(1,K_O);
  ParamN(i).beta0 = 0;
end

if rare_exist
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
end
    
a_u = 0.2;
%{
DoCV(data_raw,K_O,data_p,K_P,data_GTT,label,...
active_set_normal,active_set_rare,true,pos,mode,rare_exist,weighted);
%}
disp(a_u);
if weighted
  WN = 1;WR = 1;
end
%0:MLE regularization or 1:L2 regularization or 2:proposed model or 3: min_ent 
%if pval4R==false
[ParamPR ParamN ParamR] = ...
gradDes(data_raw,data_p,data_GTT,label,a_u,ParamPR,ParamN,ParamR,...
	active_set_normal,active_set_rare,...
	WN,WR,pos,mode,rare_exist);
%{
else
  [ParamPR ParamN ParamR] = ...
  gradDes(data_p,data_p,data_GTT,label,a_u,ParamPR,ParamN,ParamR,...
	  active_set_normal,active_set_rare,WN,WR,pval4R,mode);	
end
%}
%test PMF based on current parameters
[test_PMFunknown test_PMFnormal test_PMFrare] = ...
getTestPMF(test_raw,test_p,test_GTT,ParamPR,ParamN,ParamR,...
	   active_set_normal,active_set_rare,rare_exist);
    
%error rate on test set
[error FAR FNR error_CN error_CR error_avgC] = ...
calculate_error(test_PMFunknown,test_PMFnormal,test_PMFrare,...
		test_GTT,0,rare_exist);
disp(error_avgC);

%sparsity measure
if rare_exist
  SPARSITY.ParamPR = ParamPR;
  SPARSITY.PR_mag = norm(ParamPR.alpha);
  SPARSITY.PR_sparse = sum(ParamPR.alpha==0)/length(ParamPR.alpha);
  SPARSITY.ParamR = ParamR(active_set_rare==1);
  SPARSITY.active_set_rare = active_set_rare;
  ERROR.FAR = FAR; ERROR.FNR = FNR;
  ERROR.error_CN = error_CN; ERROR.error_CR = error_CR;
  %test set performance measure
  ROC_AUC = calculate_ROC(test_PMFunknown,test_GTT);
  %display ROC_AUC
  disp(ROC_AUC);
else 
     ROC_AUC = 0;
end

SPARSITY.ParamN = ParamN(active_set_normal==1);
SPARSITY.active_set_normal = active_set_normal;
ERROR.error = error;
ERROR.avgC = error_avgC;    
