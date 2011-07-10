function samples = discreternd(num_samples, probs)
%DISCRETERND draw samples from a discrete probability distribution
%
% samples = discreternd(num_samples, probs)
%
% Note that this function uses the obvious/naive cumulative algorithm, which is
% O(KN). For intensive use, find an implementation of a better algorithm. See
% Devroye's book
%     http://cg.scs.carleton.ca/~luc/rnbookindex.html
% for some better algorithms. Or try the GNU Scientific Library (GSL).
%
% Inputs:
%     num_samples 1x1 
%           probs Kx1 (will be normalized by this routine)
%
% Outputs:
%         samples Nx1 Discrete labels from 1..K

% Iain Murray, November 2007

%this is wayyy faster
samples = sample(probs,num_samples);
% if false && exist('octave_config_info')
%     % Remove this version when Octave has a decent histc implementation.
%     probs = probs(:)'/sum(probs);
%     samples = 1 + sum(bsxlt(cumsum(probs), rand(num_samples, 1)), 2);
%     cum_probs = cumsum(probs(:)/sum(probs));
%     [tots, samples] = histc(rand(num_samples, 1), [0;cum_probs]);
 end

