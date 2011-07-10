function log_evidence = ldae_dumb_exact(words, topics, topic_prior)
%LDAE_DUMB_EXACT evidence of LDA model (dumb code: will do *very* short documents only)
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

% Work out constant outside of sum
const = gammaln(topic_alpha) - gammaln(Nd + topic_alpha) ...
        - sum(gammaln(topic_prior));

% Pre-compute some tables for use later
gamma_terms = gammaln(bsxfun(@plus, topic_prior(:)', (0:Nd)')); % (Nd+1) x T
log_topics = log(topics);

% Explicitly create all topic assignments (at once, in memory, super dumb)
zz = nchoosek_owr(1:T, Nd, true); % (T^Nd) x Nd
Nk = histc(zz', 1:T)'; % (T^Nd) x T

% Work out big sum
terms = zeros(T^Nd, 1);
for t = 1:Nd
    terms = terms + log_topics(zz(:, t), words(t));
end
for k = 1:T
    terms = terms + gamma_terms(Nk(:, k)+1, k);
end

log_evidence = const + logsumexp(terms);
