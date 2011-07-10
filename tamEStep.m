function [theta q_a out_counts sigma score lv_score word_score] = tamEStep(x,beta,alpha,a_log_prior,sigma)
[W K A] = size(beta);
mydudes = x>0;
myx = x(mydudes);
mybeta = beta(mydudes,:,:);

if nargin == 4, sigma = ones(1,K) * (sum(x) / K) + alpha; end
if isscalar(alpha) alpha = alpha * ones(1,K); end

ran_mex_successfully = false;
if exist('tamEStepInnerMex') == 3 %if we have this mex file
    [sigma q_a ecounts score word_score lv_score] = tamEStepInnerMex(full(myx)',mybeta,alpha,a_log_prior,sigma);
    %sometimes the mex version returns NaN, i haven't been able to debug
    %this. so when it happens, i just back off to the non-mex version
    ran_mex_successfully = (max(max(max(isnan(ecounts))))==0); 
end
if ~ran_mex_successfully
    [sigma q_a ecounts score word_score lv_score] = tamEStepNoMex(myx,mybeta,alpha,a_log_prior,sigma);
end
out_counts = zeros(W,K,A);
out_counts(mydudes,:,:) = ecounts;
theta = normalize_rows(sigma);
end