% Small synthetic test problem. Should be able to get roughly correct answers
% with any method worth considering.

rand('state', 0);
randn('state', 0);

T = 3;
V = 5;
Nd = 7;
topics = rand(T, V);
topics = bsxfun(@rdivide, topics, sum(topics, 2));
topic_prior = rand(T, 1);
topic_prior = 1 * topic_prior / sum(topic_prior);
words = ceil(rand(1, Nd) * V);

exact = ldae_dumb_exact(words, topics, topic_prior)
hm = ldae_hm(words, topics, topic_prior, 1000)
bad_discretize = ldae_by_discretization(words, topics, topic_prior) % bad!
good_discretize = ldae_by_discretization_base(words, topics, topic_prior, 1000000000)
chibms = ldae_chibms(words, topics, topic_prior, 1000)
is_pseudopost = ldae_is_variants(words, topics, topic_prior, 1000000)
