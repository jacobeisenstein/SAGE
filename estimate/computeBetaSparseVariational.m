function eta = computeBetaSparseVariational(ecounts,eq_m,varargin)
%function [eta bound] =
%computeBetaSparseVariational(ecounts,eq_m,varargin)
% newton optimization, variational EM for tau.
[max_its verbose init_eta min_eta max_inv_tau] = ...
    process_options(varargin,'max-its',1,...
    'verbose',false,'init-eta',[],'min-eta',1e-20,...
    'max-inv-tau',1e5);
[W K] = size(ecounts); %eta = zeros(size(ecounts));

if isempty(init_eta), 
    eta = zeros(W,1); 
    eq_inv_tau = ones(size(eta));
else
    eta = init_eta;
    eta(eta.^2<min_eta.^2) = sign(eta(eta.^2<min_eta^2))*min_eta;
    eq_inv_tau = 1./(eta.^2);
end

if ~verbose, fprintf('.'); end

em_iter = newDeltaIterator(max_its,'debug',verbose,'thresh',1e-4); 

exp_eq_m = exp(eq_m);
while ~(em_iter.done)
    eta = newtonArmijo(@evalLogNormal,eta,{ecounts,exp_eq_m,eq_inv_tau},'debug',verbose==1,'alpha',.1,'max-its',10000);
    eq_inv_tau = 1./(eta.^2);
    eq_inv_tau(eq_inv_tau >= max_inv_tau) = max_inv_tau;
    em_iter = updateDeltaIterator(em_iter,eta);
end
end