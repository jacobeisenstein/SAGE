function samples = dirichletrnd(alpha, num_samples)
%DIRICHLETRND draw samples from a Dirichlet
%
% samples = dirichletrnd(alpha[, num_samples]);
%
% Inputs:
%            alpha Dx1 Dirichlet parameters
%      num_samples 1x1
%
% Outputs:
%         samples  D x num_samples

% Iain Murray, January 2009

if ~exist('num_samples', 'var')
    num_samples = 1;
end

if (~isvector(alpha)) || (length(alpha) < 2)
    error('alpha must be a vector with length >1');
end
D = length(alpha);

xx = gamrnd(repmat(alpha(:), 1, num_samples), 1);
samples = bsxfun(@rdivide, xx, sum(xx, 1));
