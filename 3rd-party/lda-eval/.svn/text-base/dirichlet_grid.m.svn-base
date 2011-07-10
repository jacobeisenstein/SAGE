function choices = dirichlet_grid(dirichlet_params, discretization, include_edges)
%DIRICHLET_GRID returns a set of points from a simplex using a dirichlet base measure
%
% points = simplex_grid(dirichlet_params, discretization[, include_edges=false]);
%
% Inputs:
%    dirichlet_params Dx1 Number of discrete points along each axis.
%      discretization 1x1 Number of discrete points along each axis.
%       include_edges 1x1 If true, allow components to be exactly zero or one.
%                         As this often causes problems, the default is false.
%
% Outputs:
%            choices  LOTS x D, It should be that all(1 == sum(choices, 2)).

% Iain Murray, January 2009

assert(isvector(dirichlet_params));
dirichlet_params = dirichlet_params(:)';
dim = length(dirichlet_params);
% If dim==1, the only point that is allowed is [1], and that disagrees with the
% include_edges default. For now, just don't allow this silly case.
assert(dim > 1);

vec = @(x) x(:);

if ~exist('include_edges', 'var')
    include_edges = false;
end

if include_edges
    ww = 1/(discretization-1);
    tics = (0:ww:1)';
else
    ww = 1/(discretization+1);
    tics = (ww:ww:(1-ww))';
end
assert(discretization == length(tics));

phis = zeros(dim, discretization);
rev = @(x) x(end:-1:1);
phi_param2 = rev(cumsum(rev([dirichlet_params(2:end), 0])));
for d = 1:dim
    phis(d, :) = betainv(tics, dirichlet_params(d), phi_param2(d));
end

choices = phis(1, :)';
cur_sum = choices;

% Build up choices one component at a time
for d = 2:(dim-1)
    prev_length = size(choices, 1);
    cur_sum = repmat(cur_sum, discretization, 1);
    next_val = (1-cur_sum) .* vec(repmat(phis(d,:), prev_length, 1));
    choices = [repmat(choices, discretization, 1), next_val];
    cur_sum = next_val + cur_sum;
end

% Final component must be chosen to make choices normalized
choices = [choices, 1-cur_sum];

if ~include_edges
    choices = max(eps, choices);
    choices = min(1-eps, choices);
end
