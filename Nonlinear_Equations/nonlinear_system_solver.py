import numpy as np
import matplotlib.pyplot as plt

class NonlinearSystemSolver(object):

    def __init__(self, tol=1e-5, max_iter=100, verbose=False):
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.iterations = []
    
    def __call__(self, f, *args, **kwargs):
        raise NotImplementedError("__call__() function not yet implemented.")