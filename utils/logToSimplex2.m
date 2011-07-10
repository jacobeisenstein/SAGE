function [phi logphi] = logToSimplex2(logphi,minlog)
%function [phi logphi] = logToSimplex2(logphi,minlog)
%given log(x), return exp(log(x)) / sum(exp(log(x))).
warning off
lognormalizer = logsumexp(logphi,2);
warning on
logphi = logphi - repmat(lognormalizer,1,size(logphi,2));
phi = exp(logphi);

%sketchy... is this getting used?
if nargin > 1, 
    logphi(isinf(logphi)) = minlog; 
end

end