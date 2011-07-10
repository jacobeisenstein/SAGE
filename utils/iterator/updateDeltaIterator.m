function iter = updateDeltaIterator(iter,val)
    iter.its = iter.its + 1;
    if iter.its > iter.max_its, iter.done = true; end
    if isempty(iter.prev), iter.prev = val; return; end
    delta_norm = norm(iter.prev - val,'fro');
    if iter.debug,
        fprintf('%d. %.3e\n',iter.its,delta_norm);
    end
    if delta_norm < iter.thresh 
        iter.done = true;
    end
    iter.prev = val;
end