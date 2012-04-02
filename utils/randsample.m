function y = randsample(n,k,j)
%function y = randsample(n,k)
%
%sample without replacement k integers from 1:n

if nargin==2, j = 1; end

[ig idx] = sort(rand(n,j));
y = idx(1:k,:);
end