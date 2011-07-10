function [e_log_beta score lv_score word_score] = computeBetaDirichlet(ecounts,eta)
% function beta = computeBetaBoring(ecounts)
%
% standard variational Bayesian treatment of E[log beta], where
% beta ~ Dirichlet(eta)
%
% E[log(beta)] = digamma(counts + eta) - digamma(sum(counts + eta))
% accepts multiple rows of ecounts
% eta can be a vector or a scalar (indicating a symmetric prior)
if nargin == 1, eta= 0; end
[K W] = size(ecounts);
if isscalar(eta), eta = repmat(eta,1,W); end

word_score = 0; lv_score = 0;
e_log_beta = zeros(size(ecounts));
for k = 1:K
    e_log_beta(k,:) = digamma(ecounts(k,:) + eta) - digamma(sum(ecounts(k,:)+eta));
    if nargout > 1
        word_score = word_score + ecounts(k,:)*e_log_beta(k,:)';
        lv_score = lv_score - kldirichlet(ecounts(k,:)+eta,eta); 
    end
end
score = lv_score + word_score;
end
