function [ phi ecounts ] = hdpEStep( x, log_pw, log_beta, a0, varargin)
%function [ phi ecounts ] = hdpEStep( x, log_pw, log_beta, a0, varargin)
[K W] = size(log_pw);
[T verbose] = process_options(varargin,'T',K,'verbose',false);

phi = normalize_rows(rand(T,K));
zeta = normalize_rows(rand(W,T));

iter = newDeltaIterator(50,'thresh',1e-4,'debug',verbose);
logpwx = log_pw * spdiag(x);
while ~iter.done
    a = 1 + x * zeta;
    b = a0 + sum(x*zeta) - cumsum(x*zeta);
    
    %todo: abstract this, reuse on e[log beta]
    denom = digamma(a+b);
    e_log_pi1 = digamma(a) - denom;
    e_log_pi2 = digamma(b) - denom;
    e_log_pi = e_log_pi1 + [0 cumsum(e_log_pi2(1:end-1))];
    
    phi = logToSimplex2((logpwx * zeta)' + repmat(log_beta',T,1));
    zeta = logToSimplex2(logpwx' * phi'  + repmat(e_log_pi,W,1));
    iter = updateDeltaIterator(iter,phi);
end

ecounts = phi' * zeta' * spdiag(x); 