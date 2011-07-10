function [log_evidence, log_ests] = ldae_hm(words, topics, topic_prior, iters, burn)
%LDAE_HM Approximate evidence for LDA by harmonic mean (terrible idea, do not use!)
%
% log_evidence = ldae_chibms(words, topics, topic_prior[, iters=1000[, burn=10]]);
%
% Inputs:
%             words 1xNd
%            topics TxV each row is a distribution over a vocabulary of size V 
%       topic_prior 1xT parameters of Dirichlet from which document topic vector is drawn
%             iters 1x1 Default: 1000
%
% Outputs:
%     log_evidence  1x1 

% Iain Murray, January 2009

% TODO I was in a horrible rush. Should be refactored into generic algorithm and
% put Gibbs operators in separate functions.

if ~exist('burn', 'var')
    burn = 10;
end
if ~exist('iters', 'var')
    iters = 1000;
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
log_topics = log(topics);

crap_initialization = false;
if ~crap_initialization
    % Assign latents to words in isolation as a simple initialization
    Nz = zeros(1, T);
    for t = 1:Nd
        pz = topics(:, words(t))' .* topic_prior;
        zz(t) = discreternd(1, pz);
        Nz(zz(t)) = Nz(zz(t)) + 1;
    end
    
    % Run some sweeps of Gibbs sampling
    for sweeps = 1:burn
        if mod(sweeps, 10) == 0
            fprintf('Burn Iters %d / %d\r', sweeps, burn);
        end
        for t = 1:Nd
            Nz(zz(t)) = Nz(zz(t)) - 1;
            pz = topics(:, words(t))' .* (Nz + topic_prior);
            zz(t) = discreternd(1, pz);
            Nz(zz(t)) = Nz(zz(t)) + 1;
        end
    end
    fprintf('\n');
else
    zz = ceil(rand(1, Nd) * T);
    Nz = histc(zz, 1:T);
end

% Gibbs sampler accumulating Harmonic Mean estimators
log_ests = zeros(1, iters);
for s = 1:iters
    if mod(s, 10) == 0
        fprintf('Iters %d / %d\r', s, iters);
    end
    log_like = 0;
    for t = 1:Nd
        Nz(zz(t)) = Nz(zz(t)) - 1;
        pz = topics(:, words(t))' .* (Nz + topic_prior);
        zz(t) = discreternd(1, pz);
        Nz(zz(t)) = Nz(zz(t)) + 1;
        log_like = log_like + log_topics(zz(t), words(t));
    end
    log_ests(s) = -log_like;
end
fprintf('\n');
log_evidence = - (logsumexp(log_ests(:)) - log(iters));
%log_ests
%keyboard

%% pointless thinning for a comparison I was doing once:
%log_ests = log_ests(1:40:end);
%log_evidence = - (logsumexp(log_ests(:)) - log(length(log_ests)));
