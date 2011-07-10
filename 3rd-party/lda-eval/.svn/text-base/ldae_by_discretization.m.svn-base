function log_evidence = ldae_by_discretization(words, topics, topic_prior)
%LDAE_BY_DISCRETIZATION numerically approximate evidence of LDA model (for few topics only)
%
% log_evidence = dumb_exact(words, topics, topic_prior);
%
% Inputs:
%             words 1xNd
%            topics TxV each row is a distribution over a vocabulary of size V 
%       topic_prior 1xT parameters of Dirichlet from which document topic vector is drawn
%
% Outputs:
%     log_evidence  1x1 

% Iain Murray, January 2009

[T, V] = size(topics);
Nd = length(words);

% Sanity checking input sizes
assert(isvector(topic_prior));
assert(T == length(topic_prior));
assert(isvector(words));

topic_alpha = sum(topic_prior);
%topic_mean = topic_prior / topic_alpha;

discretization = 100;
include_edges = false;
topic_settings = simplex_grid(T, discretization, include_edges); % LOTS X T
log_topic_settings = log(topic_settings);

terms = log_topic_settings * (topic_prior(:)-1); % LOTS x 1
for t = 1:Nd
    terms = terms + log(topic_settings*topics(:, words(t)));
end

log_volume = gammaln(T);
samples = length(terms);
const = gammaln(topic_alpha) - sum(gammaln(topic_prior));
log_evidence = const + logsumexp(terms) + log_volume - log(samples);
