function out = normalize_rows(in)
    out = scale_rows(in,1./row_sum(in));
% rowSums = sum(x,2);
% [I,J] = size(x);
% p = zeros(I,J);
% for i=1:I
%     p(i,:) = x(i,:) / rowSums(i);
% end
% 
end
