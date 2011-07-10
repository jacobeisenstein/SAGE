function [ alpha ] = fitDirichletPrior( e_log_theta, alpha )
%function [ alpha ] = fitDirichletPrior( e_log_theta, alpha )
%maximize E[log P(theta | alpha)], for 
%                   theta | alpha ~ Dirichlet(alpha)

if ~exist('alpha','var')
    alpha = normalize_vec(sum(exp(e_log_theta)));
end
[solution fX] = minimize(log(alpha)',@ll_log_alpha,100,e_log_theta);
alpha = exp(solution)';

    function [l g] = ll_log_alpha(log_alpha,e_log_theta)
        N = size(e_log_theta,1);
        alpha = exp(log_alpha');
        l = N * (gammaln(sum(alpha)) - sum(gammaln(alpha))) + (alpha - 1) * sum(e_log_theta)';
        g = N * (digamma(sum(alpha)) - digamma(alpha)) + sum(e_log_theta);
        g = transpose(-g .* alpha);
        l = -l;
    end

end

