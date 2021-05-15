import numpy as np
import matplotlib.pyplot as plt

from nonlinear_scalar_system_solver import NonlinearScalarSystemSolver

class SecantMethod(NonlinearScalarSystemSolver):

    name = "Secant Method"

    def __init__(self, tol=1e-5, max_iter=100, verbose=False):
        super(SecantMethod, self).__init__(tol=tol, max_iter=max_iter, verbose=verbose)

    def __call__(self, f, x1=0, x2=1):
        self.f = f
        self.iterations = []
        return self.solve(f=f, x1=x1, x2=x2)

    def solve(self, f, x1, x2):
        
        # initialization
        self.iterations.append(x1)
        self.iterations.append(x2)

        k = 0
        while np.abs(f(x2)) > self.tol and k < self.max_iter:
            k += 1

            if self.verbose:
                print(f"Iteration {k}: x1 = {x1}, x2 = {x2}")

            f1 = f(x1)
            f2 = f(x2)
            x = x2 - ( f2 * (x2 - x1) ) / ( f2 - f1 )
            if self.verbose:
                print(f"    x <-- {x2} - ({f2} * ({x2} - {x1}) / ({f2} - {f1})")
                print(f"          = {x2} - ({f2 * (x2 - x1)}) / ({f2 - f1})")
                print(f"          = {x}")
                print()
            
            x1, x2 = x2, x
            self.iterations.append(x)

        if self.verbose:
            print(f"f({x2}) = {f(x2)}")
        return x2

    def annotate_plot(self):
        for k, x in enumerate(self.iterations):
            plt.vlines(x, 0, self.f(x), color=self.colors[k % len(self.colors)], zorder=10)

if __name__ == "__main__":
    def f(x):
        return x**3 - 2*x - 5

    solver = SecantMethod(verbose=True)

    x0 = solver(f)
    solver.plot(-1, 5)