function log_evidence = ldae_w2_is_theta(words, topics, topic_prior, samples, burn)
%LDAE_W2_IS_THETA numerically approximate evidence of LDA model given first half of doc
%
% log_evidence = ldae_w2_is_theta(words, topics, topic_prior[, samples=1000[, burn=10]]);
%
% Gibbs sample topics for first half of document given first half of words.
% Sample posterior theta's (document-specific topic proportions) and use to
% predict second half of words.
%
% Inputs:
%             words 1xNd
%            topics TxV each row is a distribution over a vocabulary of size V
%       topic_prior 1xT parameters of Dirichlet from which document topic vector is drawn
%           samples 1x1 (roughly) number of points to evaluate prior theta
%        stochastic bool should I sample or use some deterministic scheme for lower variance?
%
% Outputs:
%     log_evidence  1x1 

% Iain Murray, January 2009

[T, V] = size(topics);
Nd = length(words);
n2 = floor(Nd/2) + 1;
n1end = n2 - 1;

if ~exist('samples', 'var')
    samples = 1000;
end
if ~exist('burn', 'var')
    burn = 10;
end

% Sanity checking input sizes
assert(isvector(topic_prior));
assert(T == length(topic_prior));
assert(isvector(words));

topic_prior = topic_prior(:)';
topic_alpha = sum(topic_prior);
%topic_mean = topic_prior / topic_alpha;

% init
Nz = zeros(1, T);
for t = 1:n1end
    pz = topics(:, words(t))' .* topic_prior;
    zz(t) = discreternd(1, pz);
    Nz(zz(t)) = Nz(zz(t)) + 1;
end
% Burn
for i = 1:burn
    for t = 1:n1end
        Nz(zz(t)) = Nz(zz(t)) - 1;
        pz = topics(:, words(t))' .* (Nz + topic_prior);
        zz(t) = discreternd(1, pz);
        Nz(zz(t)) = Nz(zz(t)) + 1;
    end
end
% Main
topic_settings = zeros(samples, T);
for i = 1:samples
    for t = 1:n1end
        Nz(zz(t)) = Nz(zz(t)) - 1;
        pz = topics(:, words(t))' .* (Nz + topic_prior);
        zz(t) = discreternd(1, pz);
        Nz(zz(t)) = Nz(zz(t)) + 1;
    end
    topic_settings(i, :) = dirichletrnd(topic_prior + Nz, 1)';
end

terms = zeros(samples, 1);
for t = n2:Nd
    terms = terms + log(topic_settings*topics(:, words(t)));
end

log_evidence = logsumexp(terms) - log(samples);

