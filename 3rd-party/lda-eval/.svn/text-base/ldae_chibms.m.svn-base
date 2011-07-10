function log_evidence = ldae_chibms(words, topics, topic_prior, ms_iters)
%LDAE_CHIBMS Approximate evidence for LDA using Murray & Salakhutdinov's Chib-style method
%
% log_evidence = ldae_chibms(words, topics, topic_prior);
%
% Inputs:
%             words 1xNd
%            topics TxV each row is a distribution over a vocabulary of size V 
%       topic_prior 1xT parameters of Dirichlet from which document topic vector is drawn
%          ms_iters 1x1 Default: 1000
%
% Outputs:
%     log_evidence  1x1 

% Iain Murray, January 2009

BURN_ITERS = 3;

if ~exist('ms_iters', 'var')
    ms_iters = 1000;
end

[T, V] = size(topics);
Nd = length(words);

% Sanity checking input sizes
assert(isvector(topic_prior));
assert(T == length(topic_prior));
assert(isvector(words));

topic_prior = topic_prior(:)';
topic_alpha = sum(topic_prior);
%topic_mean = topic_prior / topic_alpha;

% Assign latents to words in isolation as a simple initialization
Nz = zeros(1, T);
for t = 1:Nd
    pz = topics(:, words(t))' .* topic_prior;
    zz(t) = discreternd(1, pz);
    Nz(zz(t)) = Nz(zz(t)) + 1;
end

% Run some sweeps of Gibbs sampling
for sweeps = 1:BURN_ITERS
    for t = 1:Nd
        Nz(zz(t)) = Nz(zz(t)) - 1;
        pz = topics(:, words(t))' .* (Nz + topic_prior);
        zz(t) = discreternd(1, pz);
        Nz(zz(t)) = Nz(zz(t)) + 1;
    end
end

% Find local optimim to use as z^*, "iterative conditional modes"
% But don't spend forever on this, bail out if necessary
for i = 1:12
    old_zz = zz;
    for t = 1:Nd
        Nz(zz(t)) = Nz(zz(t)) - 1;
        pz = topics(:, words(t))' .* (Nz + topic_prior);
        [dummy, zz(t)] = max(pz);
        Nz(zz(t)) = Nz(zz(t)) + 1;
    end
    if ~isequal(old_zz, zz)
        break;
    end
end

% Run Murray & Salakhutdinov algorithm
zstar = zz;
log_Tvals = zeros(ms_iters, 1);
log_Tprob = @(zto, zfrom, Nzfrom) log_Tprob_base(zto, zfrom, Nzfrom, words, topics, topic_prior);
% draw starting position
ss = ceil(rand() * ms_iters);
% Draw z^(s)
for t = Nd:-1:1
    Nz(zz(t)) = Nz(zz(t)) - 1;
    pz = topics(:, words(t))' .* (Nz + topic_prior);
    zz(t) = discreternd(1, pz);
    Nz(zz(t)) = Nz(zz(t)) + 1;
end
zs = zz;
log_Tvals(ss) = log_Tprob(zstar, zz, Nz);
% Draw forward stuff
for sprime = (ss+1):ms_iters
    for t = 1:Nd
        Nz(zz(t)) = Nz(zz(t)) - 1;
        pz = topics(:, words(t))' .* (Nz + topic_prior);
        zz(t) = discreternd(1, pz);
        Nz(zz(t)) = Nz(zz(t)) + 1;
    end
    log_Tvals(sprime) = log_Tprob(zstar, zz, Nz);
end
% Draw backward stuff
for sprime = (ss-1):-1:1
    for t = Nd:-1:1
        Nz(zz(t)) = Nz(zz(t)) - 1;
        pz = topics(:, words(t))' .* (Nz + topic_prior);
        zz(t) = discreternd(1, pz);
        Nz(zz(t)) = Nz(zz(t)) + 1;
    end
    log_Tvals(sprime) = log_Tprob(zstar, zz, Nz);
end
% Final estimate
Nkstar = histc(zstar, 1:T); Nkstar = Nkstar(:)'; % 1xT
log_pz = sum(gammaln(Nkstar + topic_prior)) + gammaln(topic_alpha) ...
        - sum(gammaln(topic_prior)) - gammaln(Nd + topic_alpha);
log_w_given_z = 0;
for t = 1:Nd
    log_w_given_z = log_w_given_z + log(topics(zstar(t), words(t)));
end
log_joint = log_pz + log_w_given_z;
log_evidence = log_joint - (logsumexp(log_Tvals) - log(ms_iters));



function lp = log_Tprob_base(zto, zfrom, Nz, words, topics, topic_prior)
Nd = length(words);
lp = 0;
for t = 1:Nd
    Nz(zfrom(t)) = Nz(zfrom(t)) - 1;
    pz = topics(:, words(t))' .* (Nz + topic_prior);
    pz = pz/sum(pz);
    lp = lp + log(pz(zto(t)));
    Nz(zto(t)) = Nz(zto(t)) + 1;
end

