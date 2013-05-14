function [ idx val ] = sortidxs( x, dim, mode, topk, varargin )
[lo_thresh, hi_thresh] = process_options(varargin,'lo-thresh',-Inf,'hi-thresh',Inf);
%function [ idx val ] = sortidxs( x, dim, mode, topk )
if nargin < 2, 
    dim = 2;
end
if nargin < 3
    mode = 'descend';
end
[val idx] = sort(x,dim,mode);
if nargin >= 4
    idx = idx(1:topk);
    val = val(1:topk);
end
idx = idx(val < hi_thresh & val > lo_thresh);
val = val(val < hi_thresh & val > lo_thresh);
end

