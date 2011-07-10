function iter = newDeltaIterator(max_its,varargin)
    iter.max_its = max_its;
    [iter.thresh iter.min_its iter.debug] = process_options(varargin,'thresh',1e-6,'min_its',0,'debug',false);
    iter.prev = [];
    iter.done = false;
    iter.its = 0;
end