function [PMFunknown PMFnormal PMFrare] = ...
	 getTestPMF(test_raw,test_p,test_GTT,ParamPR,ParamN,ParamR,...
		    active_set_normal,active_set_rare,rare_exist)

M = size(test_raw,1);

if rare_exist
  PMFunknown = getF(-test_p,ParamPR);
  if sum(active_set_rare)>=2
    PMFrare = getF_multiclass(test_raw,test_GTT,2*ones(M,1),ParamR,active_set_rare);
  else
    PMFrare = zeros(M,length(active_set_rare));
    PMFrare(:,active_set_rare==1) = 1;
  end
else
    PMKunknown = 0;
    PMFrare = 0;
end

if sum(active_set_normal)>=2
PMFnormal = getF_multiclass(test_raw,test_GTT,2*ones(M,1),ParamN,active_set_normal);
else
    PMFnormal = zeros(M,length(active_set_normal));
    PMFnormal(:,active_set_normal==1) = 1;
end

end
    
