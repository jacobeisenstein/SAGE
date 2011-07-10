function y = randsample(n,k)
%function y = randsample(n,k)
%
%sample without replacement k integers from 1:n

[ig idx] = sort(rand(n,1));
y = idx(1:k);
end