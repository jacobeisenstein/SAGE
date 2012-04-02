function [preds model] = NBaspect(te_words,tr_words,tr_aspects,varargin)
%function preds = NBaspect(te_words,tr_words,tr_aspects)
%Naive bayes aspect prediction 
[semi_sup_frac] = process_options(varargin,'semi-sup-frac',1);
for i = 1:max(tr_aspects)
    model(i,:) = normalize_vec(sum(tr_words(tr_aspects==i,:))+sum(tr_words)); 
end
[ig preds] = max(log(model) * te_words' + repmat(digamma(hist(tr_aspects,1:max(tr_aspects))),size(te_words,1),1)');
end