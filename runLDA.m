function runLDA(K,seed,W,varargin)
%function runLDA(K,seed,W,varargin)
[debug dataset do_non_sparse do_sparse] = process_options(varargin,'debug',false,'dataset','20news','do-non-sparse',1,'do-sparse',1);

directory = sprintf('traces.lda.%s',dataset);
load(sprintf('data/%s.mat',dataset));

if strcmp(dataset,'20news')
    words = tr_data;
    [tr_words te_words widx tr_idx te_idx] = preprocess(words,'max-words',W,'debug',debug,'num-folds',10);
else
    [tr_words te_words widx] = preprocess(counts','max-words',W,'holdout',0.1,'debug',debug,'num-folds',50);
    vocab = words;
end

if ~exist(directory,'dir')
    mkdir(directory);
end

options = {'seed',seed,'te-x',te_words,'vocab',vocab(widx),'max-mstep-its',1000};
if (debug), options = cat(2,options,'max-its',10); end

basename = sprintf('%s/out.%d.%d',directory,K,seed);

if do_non_sparse
[ig theta_lda eta_lda] = sparseTAM(tr_words,K,'sparse',0,'compute-perplexity',1000,options{:});
end
if do_sparse
[ig theta_sage eta_sage] = sparseTAM(tr_words,K,'sparse',1,'compute-perplexity',50,options{:});
end
save(sprintf('%s.final.mat',basename));
end
