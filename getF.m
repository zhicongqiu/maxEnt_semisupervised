function Fs = getF(data,ParamPR)

%p-value feature, use - sign
temp = exp(ParamPR.alpha0-data*ParamPR.alpha');
Fs = temp./(1+temp);

%avoid assigning 1 or 0
Fs(Fs>1-1e-10) = 1-1e-10;
Fs(Fs<1e-10) = 1e-10;
