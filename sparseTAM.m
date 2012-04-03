function [acc theta eta_t eta_a] = sparseTAM(x,K,varargin)
%function [acc theta eta_t eta_a] = sparseTAM(x,K,varargin)
%[sparse aspects alpha gamma_dirichlet te_x te_aspects vocab ...
%    seed hyperprior_update_interval compute_perplexity ...
%    topic_aspects verbose init_eta_t init_eta_a max_its max_mstep_its ...
%    save_prefix eval_period] = ...
%    process_options(varargin,'sparse',1,'aspects',ones(size(x,1),1),'alpha',1/K,...
%    'init-gamma-dirichlet',1,'te-x',[],'te-aspects',[],...
%    'vocab',[],'seed',[],'hyperprior-update-interval',1,...
%    'compute-perplexity',1,'topic-aspects',0,'verbose',1,'init-eta-t',[],...
%    'init-eta-a',[],'max-its',100,'max-mstep-its',100,'save-prefix',[],'eval-period',5);
[sparse aspects alpha gamma_dirichlet te_x te_aspects vocab ...
    seed hyperprior_update_interval compute_perplexity ...
    topic_aspects verbose init_eta_t init_eta_a max_its max_mstep_its ...
    save_prefix eval_period] = ...
    process_options(varargin,'sparse',1,'aspects',ones(size(x,1),1),'alpha',1/K,...
    'init-gamma-dirichlet',1,'te-x',[],'te-aspects',[],...
    'vocab',[],'seed',[],'hyperprior-update-interval',1,...
    'compute-perplexity',1,'topic-aspects',0,'verbose',1,'init-eta-t',[],...
    'init-eta-a',[],'max-its',100,'max-mstep-its',100,'save-prefix',[],'eval-period',5);
disp(varargin);

%screen words that have zero counts, because they mess stuff up
word_counts = sum(x); x = x(:,word_counts>0);
if ~isempty(te_x), te_x = te_x(:,word_counts>0); end
if ~isempty(vocab), vocab = vocab(word_counts > 0); end

alpha = alpha * ones(1,K); acc = 0;
[D W] = size(x); theta = zeros(D,K); A = max(aspects);
if A > 1, compute_perplexity = 0; end

%% corpus initialization
if ~isempty(seed), rand('seed',seed); end

%mean
m = makeLogProbs(sum(x))';
%eta_t (topics)
if sparse
    if ~isempty(init_eta_t),eta_t = init_eta_t; else
        eta_t = zeros(W,K);
        if K>1, for k = 1:K,
                eta_t(:,k) = computeBetaSparseVariational(full(sum(x(randsample(D,10),:))'),m,'max-its',max_mstep_its,'verbose',0);
            end; end
    end
    %eta_a (aspects)
    if ~isempty(init_eta_a), eta_a = init_eta_a; else
        eta_a = zeros(W,A);
        if A>1
            for j = 1:A
                eta_a(:,j) = computeBetaSparseVariational(full(sum(x(aspects==j,:))'),m,'max-its',max_mstep_its);
            end
        end
    end
    %eta_ta (interactions)
    eta_ta = zeros(W,K,A);  if topic_aspects && A>1, for j = 1:A, for k = 1:K, eta_ta(:,k,j) = 0.1 * (eta_t(:,k) + eta_a(:,j)); end; end; end
    %sums
    eta_sum = makeEtaSum(m,eta_a,eta_t,eta_ta);
else
    assert(A==1); %can't do non-sparse aspect models now
    for k = 1:K
        eta_sum(:,k) = computeBetaDirichlet(full(sum(x(randsample(D,10),:))),gamma_dirichlet);
    end
    gamma_dirichlet = gamma_dirichlet * ones(1,W);
end

sigma = repmat(sum(x,2)/K,1,K) + repmat(alpha,D,1);
q_a = zeros(D+1,max(aspects)); q_a(end,:) = 1/max(aspects); %prior

eta_lv_score = 0; eta_ta_lv_score = 0; prior_prob = 0;
%iter = newIterator(max_its,'debug',true,'thresh',1e-5);
iter = newDeltaIterator(max_its,'debug',true,'thresh',0);
ecounts = zeros(W,K,A);
while ~iter.done
    
    %% E-step
    word_score = 0; estep_lv_score = 0;
    old_ecounts = ecounts;
    ecounts = zeros(W,K,A);
    for i = 1:D
        if (rem(i,100)==0), fprintf('+'); end
        old_sigma = sigma(i,:);
        log_p_a = digamma(sum(q_a)) - digamma(sum(sum(q_a)));
        
        %[theta(i,:) q_a(i,:) new_counts sigma(i,:) score doc_lv_score
        %doc_word_score] = tamEStep(x(i,:),eta_sum,alpha,log_p_a,old_sigma);
        [theta(i,:) q_a(i,:) new_counts sigma(i,:) score doc_lv_score doc_word_score] = tamEStep(x(i,:),eta_sum(:,:,aspects(i)),alpha,0,old_sigma);
        q_a(i,:) = 0; q_a(i,aspects(i)) = 1;
        ecounts(:,:,aspects(i)) = ecounts(:,:,aspects(i)) + full(new_counts);
        doc_lv_score = doc_lv_score + digamma(sum(q_a(:,aspects(i)))) - digamma(sum(sum(q_a))); %E[log P(a_i)] - E[log Q(a_i)]
        
        word_score = word_score + doc_word_score;
        estep_lv_score = estep_lv_score + doc_lv_score;
    end
    fprintf('fro norm of counts: %.3f\n',norm(reshape(ecounts,W,K*A)-reshape(old_ecounts,W,K*A),'fro'));
    
    estep_lv_score = estep_lv_score - kldirichlet(sum(q_a),q_a(end,:));
    fprintf('\n');
    computeScore(word_score,estep_lv_score,eta_lv_score,prior_prob,'print',1);
    
    %% M-step.
    if A>1 && K > 1, max_its = 10; else max_its = 0; end
    mstep_iter = newIterator(max_its,'thresh',1e-5,'debug',true);
    if sparse
        eta_t_lv_score = zeros(K,1); eta_a_lv_score = zeros(A,1);
        old_eta_t = eta_t;
        while ~mstep_iter.done
            if K > 1,
                for k = 1:K
                    eq_m = logNormalizeRows(reshape(eta_sum(:,k,:),W,A)' - repmat(eta_t(:,k),1,A)');
                    eta_t(:,k) = computeBetaSparseVariational(reshape(ecounts(:,k,:),W,A),eq_m','max-its',max_mstep_its); 
                    eta_sum = makeEtaSum(m,eta_a,eta_t,eta_ta);
                end
            end
            fprintf(' ');
            if A > 1,
                for j = 1:A
                    if sparse
                        eq_m = logNormalizeRows(eta_sum(:,:,j)' - repmat(eta_a(:,j),1,K)');
                        eta_a(:,j) = computeBetaSparseVariational(ecounts(:,:,j),eq_m','max-its',max_mstep_its);
                        eta_sum = makeEtaSum(m,eta_a,eta_t,eta_ta);
                    else
                        assert(K==1);
                        eta_a(:,j) = computeBetaDirichlet(ecounts(:,1,j)');
                    end
                end
            end
            fprintf(' ');
            if topic_aspects, for k = 1:K, for j=1:A,
                        assert(sparse==1);
                        eq_m = logNormalizeRows(eta_sum(:,k,j)' - eta_ta(:,k,j)');
                        eta_ta(:,k,j) = computeBetaSparseVariational(ecounts(:,k,j),eq_m','max-its',max_mstep_its);
                    end; end;
                eta_sum = makeEtaSum(m,eta_a,eta_t,eta_ta);
            end
            word_score = 0;
            for j = 1:A, for k = 1:K
                    word_score = word_score + scoreWords(ecounts(:,k,j),eta_sum(:,k,j));
                end; end
            eta_lv_score = sum(eta_t_lv_score) + sum(eta_a_lv_score) + sum(sum(eta_ta_lv_score));
            fprintf('\n');
            total_score = computeScore(word_score,estep_lv_score,eta_lv_score,prior_prob,'print',0);
            density_t = sum(sum(abs(eta_t) > 1e-4)) / numel(eta_t);
            density_a = sum(sum(abs(eta_a) > 1e-4)) / numel(eta_a);
            density_ta = sum(sum(sum(abs(eta_ta)>1e-4))) / numel(eta_ta);
            fprintf('%.3f\tT=%.3f A=%.3f TA=%.3f norm diff=%.3f\n',total_score,density_t,density_a,density_ta,norm(eta_t-old_eta_t,'fro'));
            %mstep_iter = updateIterator(mstep_iter,word_score + eta_lv_score);
            mstep_iter = updateDeltaIterator(mstep_iter,reshape(eta_sum,W,A*K));
        end
    else
        if A == 1
            eta_sum(:,:,1) = computeBetaDirichlet(ecounts(:,:,1)',gamma_dirichlet)';
        elseif K == 1
            for a = 1:A
                eta_sum(:,1,a) = computeBetaDirichlet(ecounts(:,1,a)',gamma_dirichlet);
            end
        else
            error('in non-sparse model, either K or A must be 1');
        end
    end
    
    %% status
    %iter = updateIterator(iter,total_score);
    iter = updateDeltaIterator(iter,[reshape(theta,D*K,1)]);% reshape(q_a(1:end-1),D*A,1)]);
    if ~isempty(vocab)
        if sparse
            if K > 1
                fprintf(' ----- topics ----- \n');
                if topic_aspects %print interaction terms
                    for k = 1:K
                        for j = 1:A
                            fprintf('K%dA%d\t ',k,j);
                            %makeTopicReport(eta_sum(1:numel(vocab),k,j)',vocab,'N',10,'background',m'); %i'm not sure supplying the background is a good idea
                            makeTopicReport(eta_ta(1:numel(vocab),k,j)',vocab,'N',10); %i'm not sure supplying the background is a good idea
                        end
                        fprintf('\n');
                    end
                end
                %same as above (for topics)
                makeTopicReport(tprod(mean(q_a)',[-3],eta_sum,[1 2 -3],'n')',vocab,'N',20);
            end
            if A > 1
                fprintf(' ----- aspects ----- \n');
                makeTopicReport(eta_a(1:numel(vocab),:)',vocab,'N',20,'background',zeros(1,W));
            end
        else
            if A > K
                for k = 1:K,
                    if K>1, fprintf(' ----- topic %d ----- ',k); end
                    makeTopicReport(reshape(eta_sum(1:numel(vocab),k,:),W,A)',vocab,'N',20);
                end
            else
                for a = 1:A
                    if A > 1, fprintf(' ----- aspect %d ----- \n', a); end
                    makeTopicReport(eta_sum(1:numel(vocab),:,a)',vocab,'N',20);
                end
            end
        end
    end
    
    %% hyperpriors
    prior_prob = 0;
    if rem(iter.its,hyperprior_update_interval) == 0
        %topics might be too sparse? consider removing this.
        if K > 1
            e_log_theta = digamma(sigma) - repmat(digamma(sum(sigma,2)),1,K);
            alpha = fitDirichletPrior(e_log_theta);
        end
        if verbose > 0, fprintf('new alpha: %s\n',sprintf('%.2f ',alpha)); end
        if ~sparse
            %gamma = fitDirichletPrior(reshape(eta_sum,W,K*A)');
            %here is a newton step that will increase your bound...
            for i = 1:10
                g_gamma = K * A * W * (digamma(W*gamma_dirichlet(1)) - digamma(gamma_dirichlet(1))) + sum(sum(sum(eta_sum)));
                g_gamma = g_gamma - 2/gamma_dirichlet(1) + 1/(gamma_dirichlet(1)^2); %from IG(1,1) prior
                h_gamma = K * A * W * (W * trigamma(W*gamma_dirichlet(1)) - trigamma(gamma_dirichlet(1)));
                h_gamma = h_gamma + 2/(gamma_dirichlet(1)^2) - 2/(gamma_dirichlet(1)^3); %from IG(1,1) prior
                gamma_dirichlet = gamma_dirichlet - 0.2 * g_gamma / h_gamma;
            end
            prior_prob = prior_prob - 2 * log(gamma_dirichlet(1)) - 1 / gamma_dirichlet(1); %from IG(1,1) prior
            if verbose > 0, fprintf('new mean gamma = %.5f\n',mean(gamma_dirichlet)); end
        end
    end
    
    %% TEST SET
    if ~isempty(te_x)
        theta_te = zeros(size(te_x,1),K);
        qa_te = zeros(size(te_x,1),max(aspects));
        e_log_a = digamma(sum(q_a)) - digamma(sum(sum(q_a)));
        
        %use the Chib estimator from Wallach et al
        if rem(iter.its,compute_perplexity)==0
            if sparse, pred_topics = logToSimplex2(eta_sum');
            else, pred_topics = logToSimplex2(computeBetaDirichlet(ecounts',gamma_dirichlet)); end
            fprintf('PERPLEX %d: %.5f\n',iter.its,computePerplexity(te_x,pred_topics,alpha));
        end
        if A > 1 && rem(iter.its,eval_period)==0
            for i = 1:size(te_x,1)
                [theta_te(i,:) qa_te(i,:)] = tamEStep(te_x(i,:),eta_sum(:,:,1:max(aspects)),alpha,e_log_a);
            end
            [ig preds] = max(qa_te');
            acc = mean(preds==te_aspects);
            fprintf('ACC %d: = %.3f\n',iter.its,acc);
        end
    end
    if ~isempty(save_prefix) && rem(iter.its,5)==1
        save(sprintf('%s.%d.mat',save_prefix,iter.its));
    end
end
if ~isempty(te_x)
    if compute_perplexity
        if sparse, pred_topics = normalize_rows(exp(eta_sum'));
        else, pred_topics = logToSimplex2(computeBetaDirichlet(ecounts',gamma_dirichlet)); end
        fprintf('FINAL PERPLEX\t%.5f\t%.5f\n',computePerplexity(te_x,pred_topics,alpha),iter.prev(1));
    end
    if A>1 && ~isempty(te_aspects )
        for i = 1:size(te_x,1)
            [theta_te(i,:) qa_te(i,:)] = tamEStep(te_x(i,:),eta_sum(:,:,1:max(aspects)),alpha,e_log_a);
        end
        [ig preds] = max(qa_te');
        acc = mean(preds==te_aspects);
        fprintf('FINAL ACC\t%.5f\t%.5f\n',acc,iter.prev(1));
    end
end
if ~sparse %can't remember why i was doing this...
    eta_t = eta_sum(:,:,1) - repmat(m,1,K);
    eta_a = eta_sum(:,1,:) - repmat(m,1,A);
end
end

function eta_sum = makeEtaSum(m,eta_a,eta_t,eta_ta)
[W K A] = size(eta_ta);
eta_sum = zeros(W,K,A);
for j = 1:A
    for k = 1:K
        eta_sum(:,k,j) = logNormalizeVec(m + eta_a(:,j) + eta_t(:,k) + eta_ta(:,k,j));
    end
end
end

function total_score = computeScore(word_score, estep_lv_score, eta_lv_score, prior_score, varargin)
[print] = process_options(varargin,'print',0);
total_score = word_score + estep_lv_score + eta_lv_score + prior_score;
if print,
    fprintf('%.3f\t=\t%.3f\t+\t%.3f\t+\t%.3f\t+\t%.3f\n',total_score,word_score,estep_lv_score,eta_lv_score,prior_score);
end
end