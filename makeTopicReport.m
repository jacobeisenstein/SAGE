function makeTopicReport(eta,vocab,varargin)
%function makeTopicReport(eta,vocab,varargin)
    [N background] = process_options(varargin,'N',10,'background',mean(eta));
    for i = 1:size(eta,1)
        if (size(eta,1)>1), fprintf('%d. ',i); end
        num_legit = sum(abs(eta(i,:) - background)>1e-3);
        fprintf('%s ',vocab{sortidxs(eta(i,:) - background,2,'descend',min(N,num_legit))})
        fprintf('\n');
    end
end