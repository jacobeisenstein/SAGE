function logp = logNormalizeVec(logp)
%function logp = logNormalizeRows(logp)
logp = log(normalize_vec(exp(logp)));
end