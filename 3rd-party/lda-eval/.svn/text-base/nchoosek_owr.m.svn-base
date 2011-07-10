function choices = nchoosek_owr(nn, kk, force_set)
%NCHOOSEK_OWR like NCHOOSEK but k ordered choices with replacement
%
%     choices = nchoosek_owr(nn, kk[, force_set=0])
%
% If nn is a non-negative integer scalar, return the number of choices, nn^kk
%
% If nn is a vector, give all possible sets of length-kk choices from this vector.
% The choices matrix is (nn^kk x kk).
%
% This style of function overloading is dangerous. Note what happens if your
% 'set' is coincidentally of length 1. This problem is rife in Matlab code. You
% can avoid bugs caused by this "feature" by setting the optional third argument
% to be non-zero. This forces the first argument to be interpreted as a set,
% even if it is a single non-negative integer.

% Iain Murray, October 2008, January 2009

if ~exist('force_set', 'var')
    force_set = 0;
end

if isscalar(nn) && (~force_set) && (round(nn) == nn) && (nn >= 0)
    choices = nn^kk; % Note: equal to 1 for kk=0
elseif isvector(nn)
    if kk == 0
        choices = zeros(1, 0); % There is one choice, which is the empty set
    else
        nn = nn(:);
        choices = nn;
        num = length(choices);
        vec = @(x) x(:);
        for ii = 1:(kk-1)
            prev_length = size(choices, 1);
            choices = [vec(repmat(nn', prev_length, 1)), repmat(choices, num, 1)];
        end
    end
else
    error('nn must be a scalar or a vector');
end

