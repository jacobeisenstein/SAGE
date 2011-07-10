function [train test] = holdoutWords(data,proportion)
%function [train test] = holdoutWords(data)
%takes sparse term-count representation, holds out 50% of words for training, 50% for test

if nargin == 1, proportion = 0.5; end

[D W] = size(data);

[rows cols vals] = find(data);
beta_param = proportion * ones(size(vals));
proportions = randbeta(beta_param, 1- beta_param);
train = sparse(rows,cols,round(vals .* proportions));
test = sparse(rows,cols,round(vals .* (1-proportions)));

end