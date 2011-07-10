function [x fx] = newton(func,init,arguments,varargin)
%function [x fx] = newton(func,init,arguments,varargin)
% do a newton optimization
% func must return the likelihood and the value of the update -H^{-1}g

[max_its init_alpha thresh debug min_alpha] = process_options(varargin,'max-its',100,'alpha',0.1,'thresh',1e-4,'debug',0,'min-alpha',1e-5);
its = 1;
x = init;
alpha = init_alpha;
while its < max_its
    [prev_score gradient step] = feval(func,x,arguments{:});
    new_score = feval(func,x + alpha * step,arguments{:});
    if debug, fprintf('%d: %.3f (alpha = %.3f)\n',its,new_score,alpha); end
    if new_score < prev_score, alpha = alpha * 1.5; end
    while (new_score > prev_score && alpha > min_alpha)
        alpha = 0.5 * alpha;
        new_score = feval(func,x + alpha * step,arguments{:});
        if debug, fprintf('%d: %.3f (alpha = %.3f)\n',its,new_score,alpha); end
    end
    x = x + alpha * step;
    fx(its) = new_score;
    its = its + 1;
    if prev_score  - new_score < thresh
        break;
    end
end
end