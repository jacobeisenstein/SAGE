function [l g step] = evalLogNormal(eta,counts,exp_eq_m,invsigsq)
Ck = sum(counts,1); C = sum(Ck); [W K] = size(counts);
denom = repmat(exp(eta),1,K).*exp_eq_m;
l = -(sum(eta' * counts) - Ck * log ( sum(denom) )' - 0.5 * trace(eta' * spdiag(invsigsq) * eta));
if nargout > 1
    beta = Ck * normalize_rows(denom') / (C + 1e-10);  %expected beta
    g = -(sum(counts,2) - C * beta' - invsigsq .* eta);
    if nargout > 2
        avec = -1./ (C * beta' + invsigsq);
        a_times_g = (-g) .* avec;
        c_times_a_times_beta = C * avec .* beta';
        %step = -a_times_g + c_times_a_times_beta ./ (1 + beta * c_times_a_times_beta) * (beta * a_times_g);
        step = -a_times_g + c_times_a_times_beta .* (beta * a_times_g ./ (1 + beta * c_times_a_times_beta));
    end
end
end

%these are slower
%l = -(sum(eta' * counts) - Ck * log ( exp(eta)' * exp(eq_m) )' - 0.5 * sum(eta.*eta.*invsigsq));
%l = -(sum(eta' * counts) - Ck * log ( exp(eta)' * exp(eq_m) )' - 0.5 * (eta.^2)'*invsigsq);
