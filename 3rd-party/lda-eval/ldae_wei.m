function log_evidence = ldae_wei(words, topics, topic_prior, samples)
%LDAE_WEI numerically approximate evidence of LDA model (for few topics only)
%
% log_evidence = ldae_wei(words, topics, topic_prior[, samples=1000]);
%
% Inputs:
%             words 1xNd
%            topics TxV each row is a distribution over a vocabulary of size V 
%       topic_prior 1xT parameters of Dirichlet from which document topic vector is drawn
%           samples 1x1 default 1000
%
% Outputs:
%     log_evidence  1x1 

% Iain Murray, January 2009

if ~exist('samples', 'var')
    samples = 1000;
end

stochastic = true;
log_evidence = ldae_by_discretization_base(words, topics, topic_prior, samples, stochastic);
