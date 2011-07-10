function y = spdiag(x)
y = sparse(1:numel(x),1:numel(x),x);
end