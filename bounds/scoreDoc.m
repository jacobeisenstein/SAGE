function [bound word_score lv_score] = scoreDoc(counts,beta,phi,x,sigma,alpha,e_log_theta)
word_score = scoreWords(counts,beta);
if nargin < 7,
    e_log_theta = digamma(sigma) - digamma(sum(sigma));
end
if numel(alpha) == 1 && numel(sigma) > 1, alpha = repmat(alpha,1,numel(sigma)); end
lv_score = e_log_theta * phi' * x';% ... %E[log p(z | theta)]
lv_score = lv_score - x * sum(log(phi).*phi,2); %E[log q(phi)]
lv_score = lv_score + gammaln(sum(alpha)) - sum(gammaln(alpha)) + e_log_theta * (alpha' - 1); % ... %E[log p(theta | alpha)]
lv_score = lv_score - gammaln(sum(sigma)) + sum(gammaln(sigma)) - e_log_theta * (sigma' - 1); %E[log q(theta)]
bound = word_score + lv_score;
end