function ub = ldae_upperbound(words, topics, true_topic_dists)
%LDAE_UPPERBOUND for synthetic data use true topic dist to evaluate likelihood as an upper bound on how well we can usually do.
%
% Inputs:
%                 words Nd x 1
%                topics TxV 
%      true_topic_dists Tx1 
%
% Outputs:
%                   ub  1x1 log p(w|all params)

% Iain Murray, January 2009

[T, V] = size(topics);
Nd = length(words);

ub = sum(log(true_topic_dists' * topics(:, words)));
