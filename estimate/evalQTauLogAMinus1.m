function [l g step] = evalQTauLogAMinus1(log_a_minus_1,b,etasq)
%function [l g step] = evalQTauA(a,b,etasq)
%
%Q(tau) = Gamma(a,b)
%eta \sim N(0,tau)
%tau \sim 1/tau
%
%a is Wx1
%b is Wx1
%eta is Wx1

a = exp(log_a_minus_1) + 1;

%l = -.5 E[log tau] - .5 eta.^eta.^ E[1/tau] - E[log tau] - E[log Q(tau)]
log_b = log(b);
l = -(a + .5).*(digamma(a) + log_b) - .5 * etasq ./ ((a-1).*b) + a + gammaln(a) + a .* log_b;
g = -(a + .5).*(trigamma(a)) + .5 * etasq ./ (b .* (a-1) .* (a-1)) + 1;

l = -sum(l);
g = -g .* (a - 1);

step = 0; %do this later
if nargout > 2
    error('newton step size not yet implemented');
end

end