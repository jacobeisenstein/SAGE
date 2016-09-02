from numpy.linalg import norm
import copy

class DeltaIterator:
    def __init__(s,max_its=100,thresh=1e-4,debug=False):
        s.thresh = thresh
        s.max_its = max_its
        s.prev = None
        s.done = False
        s.its = 0
        s.debug = debug

    def update(s,x):
        if s.prev is not None:
            change = norm(x - s.prev) / (1e-6+norm(x))
            if s.debug: print s.its,'/',s.max_its,change
            if change < s.thresh: s.done = True
        s.its += 1
        if s.its > s.max_its: s.done = True
        s.prev = copy.copy(x)
        
