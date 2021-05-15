import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from difference_equation_solver import DifferenceEquationSolver

class ImplicitSolver(object):
    '''
    Abstract Class for numerically solving Implicit Ordinary Diffence Equations
    
    General Form
            u_{n+1} = a_0 u_n + a_{-1} u_{n-1} + ... + h [b_0 u'_n + ...]
    '''
    name = "Implicit Time Marching"

    def __init__(self, var='u', verbose=True, *args, **kwargs):
        super(ImplicitSolver, self).__init__(verbose=verbose, *args, **kwargs)
        self.var = sp.symbol(var)
    
    def init_time_marching(self, u0):
        raise NotImplementedError("Implement your time marching method")

    def time_marching(self, F, Stages, h, n, u_np1=None):
        raise NotImplementedError("Implement your time marching method")
    
    def solve(self, F, u0, h=None, N=None, T=None):
        h, N, T = super(ImplicitSolver, self).solve(F, u0, h=h, N=N, T=T)
        
        Stages = self.init_time_marching(u0)
        for n in range(N):
            Stages, RHS = self.time_marching(F, Stages, h, n)
            
            # solving implicit equation using non-linear methods
            

        U = Stages[0]
        TimeSteps = np.linspace(0, T, N+1)

        self.U = U
        self.TimeSteps = TimeSteps

        return U, TimeSteps