function iter = newIterator(max_its,varargin)
%function iter = newIterator(max_its,varargin)
%
% create new iterator.  assumes the value is always increasing
    iter.max_its = max_its;
    [iter.thresh iter.min_its iter.debug] = process_options(varargin,'thresh',0,'min-its',0,'debug',false);
    iter.prev = -inf;
    iter.done = false;
    iter.its = 0;
end