import numpy as np
import matplotlib.pyplot as plt

from difference_equation_solver import DifferenceEquationSolver

class ExplicitSolver(DifferenceEquationSolver):
    '''
    Abstract Class for numerically solving Explicit Ordinary Diffence Equations
    
    General Form
            u_{n+1} = a_0 u_n + a_{-1} u_{n-1} + ... + h [b_0 u'_n + ...]
            (no u'_{n+1} term)
    '''
    name = "Explicit Time Marching"

    def __init__(self, verbose=True, *args, **kwargs):
        super(ExplicitSolver, self).__init__(verbose=verbose, *args, **kwargs)

    def init_time_marching(self, u0):
        raise NotImplementedError("Implement your time marching method")

    def time_marching(self, F, Stages, h, n):
        raise NotImplementedError("Implement your time marching method")
    
    def solve(self, F, u0, h=None, N=None, T=None):
        h, N, T = super(ExplicitSolver, self).solve(F, u0, h=h, N=N, T=T)
        
        Stages = self.init_time_marching(u0)
        for n in range(N):
            Stages = self.time_marching(F, Stages, h, n)
            
        U = Stages[0]
        TimeSteps = np.linspace(0, T, N+1)

        self.U = U
        self.TimeSteps = TimeSteps

        return U, TimeSteps
    

class Test(ExplicitSolver):
    ''' Gives an Example of how to use this code '''

    def init_time_marching(self, u0):
        return [ [u0], [], [] ]
    
    def time_marching(self, F, Stages, h, n, verbose=True):
        u_n = Stages[0][-1]

        utilde_n = u_n + (h/3) * F(u_n, h*n)
        ubar_n = u_n + (h/2) * F(utilde_n, h*(n+1/3))
        u_np1 = u_n + h * F(ubar_n, h*(n+1/2))

        Stages[1].append(utilde_n)
        Stages[2].append(ubar_n)
        Stages[0].append(u_np1)

        return Stages

if __name__ == "__main__":

    def F(u, t):
        return -u + np.sin(t)
    
    eq = Test()
    u0 = 1
    eq.solve(F, u0, h=1/64, T=1)
    eq.plot()