function log_evidence = ldae_by_discretization_base(words, topics, topic_prior, samples, stochastic)
%LDAE_BY_DISCRETIZATION numerically approximate evidence of LDA model (for few topics only)
%
% log_evidence = dumb_exact(words, topics, topic_prior[, samples=1000[, stochastic=false]]);
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

if ~exist('samples', 'var')
    samples = 1000;
end
if ~exist('stochastic', 'var')
    stochastic = false;
end

% Sanity checking input sizes
assert(isvector(topic_prior));
assert(T == length(topic_prior));
assert(isvector(words));

topic_alpha = sum(topic_prior);
%topic_mean = topic_prior / topic_alpha;

if stochastic
    topic_settings = dirichletrnd(topic_prior, samples)'; % samples x T
else
    discretization = ceil(samples^(1/T));
    topic_settings = dirichlet_grid(topic_prior, discretization); % LOTS x T
end
samples = size(topic_settings, 1);
log_topic_settings = log(topic_settings);

terms = zeros(samples, 1);
for t = 1:Nd
    terms = terms + log(topic_settings*topics(:, words(t)));
end

log_evidence = logsumexp(terms) - log(samples);

