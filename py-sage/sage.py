from numpy import ones, zeros, exp, log, tile, array, dot, reshape
from scipy.optimize import minimize
#from deltaIterator import DeltaIterator
import deltaIterator as di

#eq_m = numpy.log(l.ecounts.sum(axis=1)) - scipy.misc.logsumexp(numpy.log(l.ecounts))
#sage.estimate(l.ecounts[:,0:1],eq_m)
#sage.fLogNormalAux(numpy.zeros([W,1]),l.ecounts[:,0:1],numpy.exp(eq_m).reshape(W,1),numpy.ones([W,1]))

def estimate(ecounts,eq_m,max_its=25):
    if len(ecounts.shape)==1:
        ecounts = reshape(ecounts,(-1,1))
    [W,K] = ecounts.shape
    eta = zeros(W)
    eq_inv_tau = ones(W)
    exp_eq_m = exp(eq_m)
    max_inv_tau = 1e5
    it = di.DeltaIterator(debug=False,max_its=max_its,thresh=1e-4)
    while not(it.done):
        fLogNormal = lambda x : fLogNormalAux(x,ecounts,exp_eq_m,eq_inv_tau)
        gLogNormal = lambda x : gLogNormalAux(x,ecounts,exp_eq_m,eq_inv_tau)
        min_out = minimize(fLogNormal,eta,method='L-BFGS-B',jac=gLogNormal,options={'disp':False})
        #TODO:
        #hpLogNormal = lambda x : hpLogNormalAux(x,ecounts,exp_eq_m,eq_inv_tau)
        #min_out = minimize(fLogNormal,eta,method='Newton-CG',jac=gLogNormal,options={'disp':True})
        eta = min_out.x
        eq_inv_tau = 1/(eta**2)
        eq_inv_tau[eq_inv_tau > max_inv_tau] = max_inv_tau
        it.update(eta)
    return(eta)

def fLogNormalAux(eta,ecounts,exp_eq_m,eq_inv_tau):
    C = ecounts.sum(axis=0)
    [W,K] = ecounts.shape
    denom = tile(exp(eta),(K,1)).dot(exp_eq_m.T)
    out = -(eta.T.dot(ecounts).sum(axis=0) - C * log(denom.sum(axis=0)) - 0.5 * eq_inv_tau.T.dot(eta ** 2))
    return(out[0])
           
def gLogNormalAux(eta,ecounts,exp_eq_m,eq_inv_tau):
    C = ecounts.sum(axis=0)
    [W,K] = ecounts.shape
    denom = tile(exp(eta),(K,1)) * exp_eq_m
    denom_norm = (denom.T / denom.sum(axis=1))
    beta = C * denom_norm / (C + 1e-10)
    g = -(ecounts.sum(axis=1) - beta.dot(C) - eq_inv_tau * eta)
    return(g)

# utility
def makeVocab(counts,min_count):
    N = sum([x > min_count for x in counts.values()])
    vocab = [word for word,count in counts.most_common(N)] #use vocab.index() to get the index of a word
    return vocab    

def makeCountVec(counts,vocab):    
    vec = zeros(len(vocab))
    for i,word in enumerate(vocab):
        vec[i] = counts[word]
    return vec

def topK(beta,vocab,K=10):
    return [vocab[idx] for idx in (-beta).argsort()[:K]]
    
