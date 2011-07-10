function logp = logNormalizeRows(logp)
%function logp = logNormalizeRows(logp)
%logp_old = log(normalize_rows(exp(logp)));
for k = 1:size(logp,1),
    logp(k,:) = logp(k,:) - logsumexp(logp(k,:)');
end
end