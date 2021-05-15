from imports import *

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt



if __name__ == "__main__":

    Relax = GuassSeidelRelaxation(verbose=True)

    A = Relax.B(3, (1, 2, 1))
    f = np.array([1, 2, 3])

    X0 = [1, 1, 1]
    Relax(A, f, X0=X0)
    print()
    print("Exact Solution:   ", Relax.exact_solution(A, f))

    #fd_taylor_table([-1, 0, 1], [], [-1, 1], P=2, depth=10)
    
    """
    def F(u, t):
        return -u + np.sin(t)

    eq = ExplicitEulerSolver()
    u0 = 1
    eq.solve(F, u0, h=1/64, T=1)
    eq.plot()

    print()
    
    representative_equation_implicit_euler(1, -1, 0, 4, h=0.1, N=100, T=None, verbose=True, plot=True)
    """