function runTAM(K,seed,W,varargin)
%function runTAM(K,seed,W,varargin)
% run a topic-aspect model on 20 news data (can't do this for NIPS data)
[debug samp_rate] = process_options(varargin,'debug',false,'samp-rate',1);

dataset = '20news';
directory = sprintf('traces.lda.%s',dataset);
load(sprintf('data/%s.mat',dataset));

words = tr_data;
[tr_words te_words widx tr_idx te_idx] = preprocess(words,'max-words',W,'debug',debug,'num-folds',10);

if ~exist(directory,'dir')
    mkdir(directory);
end

options = {'seed',seed,'te-x',te_words,'vocab',vocab(widx),'max-mstep-its',1000};
if (debug), options = cat(2,options,'max-its',10); end

N = numel(tr_idx);
tr_idx_samp = randsample(N,round(N*samp_rate));
sparseTAM(tr_words(tr_idx_samp,:),K,'sparse',1,'aspects',tr_aspect(tr_idx(tr_idx_samp)),'te-x',te_words,'te-aspects',tr_aspect(te_idx)','eval-period',1,'compute-perplexity',0,options{:});
end


