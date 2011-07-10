function [perplex ll ll_per_word] = computePerplexity(docs,topics,alpha,varargin)
%function [perplex ll ll_per_word] = computePerplexity(docs,topics,alpha,varargin)
%calls out to slow 3rd-party code (Wallach, Murray, Salakhutdinov, Mimno)
%for computing perplexity, the right way

[num_its ] =process_options(varargin,'num-its',100);
doc_log_prob = 0;    
fprintf('computing perplexity');
for i = 1:size(docs,1)
    doc_log_prob = doc_log_prob + ldae_chibms(termCountsToWordList(docs(i,:)),topics,alpha,num_its);
    if rem(i,10)==0, fprintf('='); end
end
fprintf('\n');
ll_per_word = doc_log_prob / sum(sum(docs));
perplex = exp(-ll_per_word);
end