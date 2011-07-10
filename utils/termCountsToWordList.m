function list = termCountsToWordList(term_counts)
%function word_list = termCountsToWordList(term_counts)
% takes a (possibly sparse) term counts matrix and returns a word list
% calls a cute mex function
for i = 1:size(term_counts,1)
    [ig j s] = find(term_counts(i,:));
    list{i} = termCountsToWordListMex(j,s);
end
if size(term_counts,1) == 1, list = list{1}; end
end

