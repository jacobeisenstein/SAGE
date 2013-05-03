function y = randsample(n,k,j)
%function y = randsample(n,k)
%
%sample without replacement k integers from 1:n
if nargin<2, k = n; end
if nargin<3, j = 1; end
if numel(n) == 1
    n = 1:n;
end
[ig idx] = sort(rand(numel(n),j));
y = n(idx(1:k,:));
end