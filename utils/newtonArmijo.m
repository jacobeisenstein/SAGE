function [x fx] = newtonArmijo(func,init,arguments,varargin)
%function [x fx] = newton(func,init,arguments,varargin)
% do a newton minimization
% func must return the likelihood and the value of the update -H^{-1}g

[max_its init_alpha thresh debug min_alpha] = process_options(varargin,'max-its',1000,'init-alpha',1,'thresh',1e-8,'debug',0,'min-alpha',1e-15);
x = init;
beta = 1e-4; tau = 0.25; 

iterator = newIterator(max_its,'thresh',thresh,'debug',debug);
alpha = init_alpha;
while ~iterator.done
    [old_score gradient step] = feval(func,x,arguments{:});
    
    %do armijo linear-searchto find step size
    new_score = feval(func,x+alpha*step,arguments{:});
    while alpha > min_alpha && (isnan(new_score) || isinf(new_score) || new_score > old_score + beta * gradient' * (alpha * step))
        %if debug, fprintf('alpha: %.2e -> %.2e\n',alpha,alpha*tau); end
        alpha = alpha * tau;
        new_score = feval(func,x+alpha*step,arguments{:});
    end
    if alpha < min_alpha, alpha == 0; end
    try %try to take a step, if you can
        x = x + alpha * step;
        alpha = alpha ./ tau; %walk back one step
        fx(iterator.its+1) = new_score;
        iterator = updateIterator(iterator,-new_score);
    catch
        iterator.done = true;
    end
end

