function s = logsumexp(a, dim)
% Returns log(sum(exp(a),dim)) while avoiding numerical underflow.
% Default is dim = 1 (columns).
% logsumexp(a, 2) will sum across rows instead of columns.
% Unlike matlab's "sum", it will not arbitrarily switch the summing direction
% if you provide a row vector.

% Written by Tom Minka

if nargin < 2
  dim = 1;
end

% subtract the largest in each column
[y, i] = max(a,[],dim);
dims = ones(1,ndims(a));
dims(dim) = size(a,dim);
a = a - repmat(y, dims);
s = y + log(sum(exp(a),dim));
i = find(~isfinite(y)); % IAM tweaked to isfinite to stop Matlab warning
if ~isempty(i)
  s(i) = y(i);
end
