function y = kldirichlet(p,q)
%function y = kldirichlet(p,q)
    sumq = sum(q); sump = sum(p);
    y = gammaln(sump) - gammaln(sumq) + sum(gammaln(q)-gammaln(p)) + ...
        (p - q) * (digamma(p) - digamma(sump))';
end