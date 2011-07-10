function score = scoreWords(counts,beta)
%function score = scoreWords(counts,beta)
    score = sum(sum(counts .* beta));% - logsumexp(beta') * sum(counts,2);
end