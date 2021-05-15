import numpy as np
import matplotlib.pyplot as plt

from nonlinear_scalar_system_solver import NonlinearScalarSystemSolver

class NewtonMethod(NonlinearScalarSystemSolver):

    name = "Newton's Method"

    def __init__(self, tol=1e-5, max_iter=100, verbose=False):
        super(NewtonMethod, self).__init__(tol=tol, max_iter=max_iter, verbose=verbose)

    def __call__(self, f, df=None, x0=0):
        if df is None:
            raise NotImplementedError("Implement SymPy automatic differentiation.")
        
        self.f = f
        self.df = df
        self.iterations = []
        return self.solve(f=f, df=df, x0=x0)

    def solve(self, f, df, x0):

        # initialization
        x = x0
        self.iterations.append(x)

        k = 0
        while np.abs(f(x)) > self.tol and k < self.max_iter:
            k += 1

            f_k = f(x)
            df_k = df(x)
            x_k = x - f_k / df_k
            if self.verbose:
                print(f"Iteration {k}: x{k-1} = {x}")
                print(f"    x <-- {x} - {f_k} / {df_k} = {x_k}")
                print()
            
            x = x_k
            self.iterations.append(x)

        if self.verbose:
            print(f"f({x}) = {f(x)}")
        return x
    
    def annotate_plot(self):
        for k, x in enumerate(self.iterations):
            plt.vlines(x, 0, self.f(x), color=self.colors[k % len(self.colors)], zorder=10)
        
        

if __name__ == "__main__":
    def f(x):
        return x**3 - 2*x - 5
    
    def df(x):
        return 3*x**2 - 2

    solver = NewtonMethod(verbose=True)

    x0 = 2
    x = solver(f, df, x0=x0)

    solver.plot(1, 3)