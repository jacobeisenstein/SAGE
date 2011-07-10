function [tr_words te_words widx tr_idx te_idx] = preprocess(counts,varargin)
%function [tr_words te_words widx tr_idx te_idx] = preprocess(counts,varargin)
[debug max_words num_folds fold holdout] = process_options(varargin,'debug',0,'max-words',5000,'num-folds',5,'fold',1,'holdout',1);
[N W] = size(counts);

counts = holdoutWords(counts,holdout);

te_idx = fold:num_folds:N;
tr_idx = setdiff_sorted(1:N,te_idx);
if debug
    tr_idx = tr_idx(randsample(numel(tr_idx),min(1000,numel(tr_idx))));
    te_idx = te_idx(randsample(numel(te_idx),min(100,numel(te_idx))));
end
count_sums = sum(counts(tr_idx,:)>0);
max_words = min(max_words,sum(count_sums>0));

widx = sortidxs(count_sums,2,'descend',max_words);
tr_idx = tr_idx(sum(counts(tr_idx,widx),2)>0);
te_idx = te_idx(sum(counts(te_idx,widx),2)>0);
tr_words = [counts(tr_idx,widx)]; %sum(ap_full(tr_idx,nonwords),2)];
te_words = [counts(te_idx,widx)]; %sum(ap_full(te_idx,nonwords),2)];

%tr_words = holdoutWords(tr_words,holdout);
%te_words = holdoutWords(te_words,holdout);
end