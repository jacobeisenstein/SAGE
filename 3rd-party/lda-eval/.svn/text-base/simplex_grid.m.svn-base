function choices = simplex_grid(dim, discretization, include_edges)
%SIMPLEX_GRID returns a discrete set of points from a simplex
%
% points = simplex_grid(dim, discretization[, include_edges=false]);
%
% Inputs:
%                 dim 1x1 dimensionality D
%      discretization 1x1 Number of discrete points along each axis.
%       include_edges 1x1 If true, allow components to be exactly zero or one.
%                         As this often causes problems, the default is false.
%
% Outputs:
%            choices  LOTS x D, It should be that all(1 == sum(choices, 2)).

% Iain Murray, January 2009

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

choices = tics;
cur_sum = choices;

% Build up choices one component at a time
for d = 2:(dim-1)
    % For each existing choice, consider setting next component to every choice
    % in tics
    prev_length = size(choices, 1);
    proposed_next = vec(repmat(tics', prev_length, 1));
    proposed_sum = proposed_next + repmat(cur_sum, discretization, 1);
    proposed_choices = [proposed_next, repmat(choices, discretization, 1)];
    % Only keep proposals that can lead to valid points on simplex
    idx = proposed_sum <= (1 - tics(1)*(dim-d));
    choices = proposed_choices(idx, :);
    cur_sum = proposed_sum(idx);
end

% Final component must be chosen to make choices normalized
choices = [choices, 1-cur_sum];
