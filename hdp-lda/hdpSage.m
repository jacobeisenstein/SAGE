function [ eta ] = hdpSage( x, Kmax, varargin )
%function [ eta ] = hdpSage( x, Kmax, varargin )
% from Wang, Paisley and Blei AISTATS 2011 section 3.1, ä new
% coordinate-ascent variational inference
[maxits a0 gamma max_mstep_its vocab] = process_options(varargin,'max-its',100,'a0',1,'gamma',1,'max-mstep-its',25,'vocab',[]);
[D W] = size(x);
m = makeLogProbs(sum(x));
ecounts= zeros(Kmax,W);
for k = 1:Kmax, ecounts(k,:) = full(sum(x(randsample(D,5),:))'); end
iter = newDeltaIterator(maxits,'thresh',1e-4);
u = ones(Kmax,1); v = gamma * ones(Kmax,1);
while ~iter.done
    %mstep
    e_log_beta1 = digamma(u) - digamma(u+v);
    e_log_beta2 = digamma(v) - digamma(u+v);
    e_log_beta = e_log_beta1 + [0; cumsum(e_log_beta2(1:end-1))];
    
    if true
        for k = 1:Kmax
            eta(k,:) = computeBetaSparseVariational(ecounts(k,:)',m','max-its',max_mstep_its,'verbose',0);
        end
        [~,log_pw] = logToSimplex2(eta + repmat(m,Kmax,1));
    else
        log_pw = makeLogProbs(ecounts + 1e-3);
        eta = log_pw - repmat(mean(log_pw),Kmax,1); %just for visualization
    end
    
    for k = 1:Kmax
        prevalence = sum(ecounts(k,:)) / sum(sum(ecounts));
        if prevalence > 1/(10*Kmax) & ~isempty(vocab)
            fprintf('%d %.3f %s\n',k,prevalence,sprintf('%s ',vocab{sortidxs(eta(k,:),2,'descend',10)}));
        end
    end
    
    ecounts = zeros(Kmax,W);
    u = ones(Kmax,1); v = gamma * ones(Kmax,1);
    for d = 1:D
        [phi dcounts] = hdpEStep(x(d,:),log_pw,e_log_beta,a0,'T',10);
        u = u + sum(phi,1)';
        v = v + (repmat(sum(sum(phi,1)),1,Kmax) - cumsum(sum(phi,1)))'; %check this
        ecounts = ecounts + dcounts;
        if rem(d,100)==0, fprintf('+'); end
    end
    fprintf('\n');
end
end

