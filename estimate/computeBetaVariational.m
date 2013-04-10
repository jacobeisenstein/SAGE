function [eta bound lv_score word_score] = computeBetaVariational(ecounts,eq_m,varargin)
%function [eta bound] =
%computeBetaVariational(ecounts,eq_m,varargin)
% newton optimization, variational EM for tau (shared across words, so not
% sparse)
[max_its verbose init_eta given_precision] = process_options(varargin,'max-its',1,'verbose',false,'init-eta',[],'precision',[]);
[W K] = size(ecounts); %eta = zeros(size(ecounts));

if isempty(init_eta) || max(abs(init_eta)) == 0, 
    eta = zeros(W,1); 
    precision = 1;
else
    eta = init_eta;
    precision = W./sum(eta.^2);
end
if ~isempty(given_precision), 
    precision = given_precision; 
    max_its = 0;
end

if verbose==0.5, fprintf('.'); end

%em_iter = newDeltaIterator(max_its,'debug',verbose,'thresh',1e-2); 
em_iter = newIterator(max_its,'debug',verbose,'thresh',1e-6);

exp_eq_m = exp(eq_m);
if sum(sum(ecounts)) == 0
    eta = zeros(W,K);
else
    while ~(em_iter.done)
        [eta fX_newton] = newtonArmijo(@evalLogNormal,eta,{ecounts,exp_eq_m,precision},'debug',verbose==1,'max-its',10000);
        bound = fX_newton(end);

        if isempty(given_precision)
            precision = W./sum(eta.^2);
            bound = bound + .5 * W * log(precision);
        end
        if isinf(bound) break; end
        em_iter = updateIterator(em_iter,bound);
    end
end
word_score = scoreWords(ecounts,logNormalizeRows(repmat(eta,1,K)'+eq_m')');
lv_score = -.5 * trace(eta' * eta) * precision;
if isempty(given_precision), lv_score = lv_score + .5 * W * log(eq_inv_tau); end 
bound = word_score + lv_score;
