function logProbs = makeLogProbs(counts,varargin)
%function logProbs = makeLogProbs(counts,varargin)
%each row is a set of counts. convert to normalized log probabilities.
[min_log_prob] = process_options(varargin,'min-log-prob',-200);
logProbs = log(full(normalize_rows(counts)));
logProbs(logProbs < min_log_prob) = min_log_prob;
logProbs = log(normalize_rows(exp(logProbs)));
end