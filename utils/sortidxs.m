function [ idx val ] = sortidxs( x, dim, mode, topk )
%function [ idx val ] = sortidxs( x, dim, mode, topk )
if nargin < 2, 
    dim = 2;
end
if nargin < 3
    mode = 'descend';
end
[val idx] = sort(x,dim,mode);
if nargin == 4
    idx = idx(1:topk);
    val = val(1:topk);
end
end

