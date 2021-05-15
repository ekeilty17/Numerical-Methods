import matplotlib.pyplot as plt

class DifferenceEquationSolver(object):
    '''
    Abstract Class for numerically solving Ordinary Diffence Equations
    
    General Form
            u_{n+1} = a_0 u_n + a_{-1} u_{n-1} + ... + h [b_0 u'_n + ...]
    '''
    name = "Time Marching"

    def __init__(self, verbose=True, *args, **kwargs):
        self.verbose = verbose
    
    def init_time_marching(self, u0):
        raise NotImplementedError("Implement your time marching method")

    def time_marching(self, F, Stages, h, n, u_np1=None):
        raise NotImplementedError("Implement your time marching method")
    
    def solve(self, F, u0, h=None, N=None, T=None):
        # derivative function F = F(u, t)
        if  ((h is None) and (N is None)) or \
            ((N is None) and (T is None)) or \
            ((T is None) and (h is None)):
            raise ValueError("Must specify 2 out of 3 of the following: time step (h), total number of steps (N), or final time (T)")
        
        if h is None:
            h = T / N
        if N is None:
            N = T / h
            if int(N) != N:
                print("Warning: h does not evenly divide T, thus the endpoint will be slightly off")
            N = int(round(N, 0))
        if T is None:
            T = h * N
        
        return h, N, T

    def plot(self, U=None, TimeSteps=None):
        
        U = self.U if U is None else U
        TimeSteps = self.TimeSteps if TimeSteps is None else TimeSteps

        plt.plot(TimeSteps, U)
        plt.grid()
        plt.xlabel("t")
        plt.ylabel("u")
        plt.title(f"{self.name.title()}")
        plt.show()