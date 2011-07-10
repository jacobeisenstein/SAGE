function iter = updateIterator( iter, val )
%function [ iter ] = updateIterator( iter, val )
iter.its = iter.its + 1;

if isnan(val)
    error('failure!');
end

change = (val - iter.prev) / abs(iter.prev);
iter.prev = val;
if iter.debug, fprintf('it: %d\tscore: %.3f\tchange: %.7f\n',iter.its,val,change); end
if (iter.its > iter.max_its || (iter.its > iter.min_its && change < iter.thresh))
    iter.done = true;
end
end

