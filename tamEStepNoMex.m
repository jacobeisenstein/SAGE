function [sigma q_a out_counts score lv_score word_score] = tamEStepNoMex(x,beta,alpha,a_log_prior,sigma)
%function [sigma q_a out_counts score lv_score word_score] = tamEStepNoMex(x,beta,alpha,a_log_prior,sigma)
%
%this contains a lot of optimizations that make it kind of complicated.
%maybe i should release a simplified version?
[W K A] = size(beta);
if nargin < 5
    if isscalar(alpha), sigma = ones(1,K) * (numel(x)  / K + alpha);
    else sigma = ones(1,K) * numel(x) / K + alpha; end
end
mydudes = x>0;
q_a = normalize_vec(ones(1,A));
myphi = 1/K * ones(sum(mydudes),K);
myx = x(mydudes);
mybeta = beta(mydudes,:,:);

%new_counts = spalloc(K,W,sum(mydudes)*K);
xrep = full(repmat(myx',1,K));
iter = newIterator(5,'thresh',1e-4,'debug',false);
%iter_delta = newDeltaIterator(1000,'thresh',1e-3);
iter_ctr = 0;
while ~iter.done
    sigma = (myx * myphi + alpha);
    dig_sig = digamma(sigma) - digamma(sum(sigma));
    warning off;
    myphi = logToSimplex2(tprod(mybeta,[1 2 -1],q_a,[3 -1],'n') + repmat(dig_sig,size(myx,2),1));
    [q_a log_q_a] = logToSimplex2(squeeze(tprod(mybeta,[-1 -2 1],myphi,[-1 -2],'n'))' + a_log_prior);
    warning on;
    %score it. this is slow, so we do it less often.
    if rem(iter_ctr,5) == 0
        word_score = 0;
        %new_counts(:,mydudes) = transpose(myphi.*xrep);
        my_new_counts = transpose(myphi.*xrep);
        %for a = 1:A
        %new_counts = new_counts * q_a(a);
        %    word_score = word_score + q_a(a) * scoreWords(my_new_counts',mybeta(:,:,a));
        %end
        %same as above, but faster
        word_score = q_a * tprod(my_new_counts,[-1 -2],mybeta,[-2 -1 1],'n');
        %using mybeta(:,:,1) looks weird, but actually this part is being ignored, we're only using the latent variable score. so this is legit.
        [ig ig2 lv_score] = scoreDoc(my_new_counts',mybeta(:,:,1),myphi,myx,sigma,alpha,dig_sig);
        lv_score = lv_score + q_a * (a_log_prior - log_q_a)';
        score = word_score + lv_score;
        
        iter = updateIterator(iter,score);
    end
    iter_ctr = iter_ctr + 1;
end
%compute output counts for real
if nargout >= 3
    out_counts = zeros(W,K,A);
    for k=1:K, for a=1:A, out_counts(mydudes,k,a) = myphi(:,k) .* myx' .* q_a(a); end; end
end
end