import numpy as np
import matplotlib.pyplot as plt

from nonlinear_scalar_system_solver import NonlinearScalarSystemSolver

class BisectionMethod(NonlinearScalarSystemSolver):

    name = "Bisection Method"

    def __init__(self, tol=1e-5, max_iter=100, verbose=False):
        super(BisectionMethod, self).__init__(tol=tol, max_iter=max_iter, verbose=verbose)

    def __call__(self, f, a=-10, b=10):
        self.f = f
        self.iterations = []
        return self.solve(f=f, a=a, b=b)

    def solve(self, f, a, b):
        
        if a >= b:
            raise ValueError("a must be strictly less than b")
        
        if f(a) * f(b) >= 0:
            raise ValueError("f(a) * f(b) must be less than 0")

        # initialization
        x = b
        self.iterations.append( (a, b) )

        k = 0
        while np.abs(f(x)) > self.tol and k < self.max_iter:
            
            x = (a + b) / 2
            k += 1

            if self.verbose:
                print(f"Iteration {k}")
                print(f"\t[a, b] = [{a}, {b}]")
                print(f"\tx = {x}")
                print()
                print(f"\tf(a) * f(x) = {f(a)} * {f(x)} = {f(a) * f(x)}", end="")

            if f(a) * f(x) < 0:
                b = x
                if self.verbose:
                    print(" < 0")
                    print(f"\tx --> b")
            else:
                if self.verbose:
                    print(" > 0")
                    print(f"\tx --> a")
                a = x

            self.iterations.append( (a, b) )
            if self.verbose:
                print()
        
        if self.verbose:
            print(f"f({x}) = {f(x)}")
        return x

    def annotate_plot(self):

        x = self.iterations[0][0]
        plt.vlines(x, 0, self.f(x), color="red")

        for k, (ak, bk) in enumerate(self.iterations, 1):
            x = ak if x == bk else bk
            plt.vlines(x, 0, self.f(x), color=self.colors[k % len(self.colors)], zorder=10)

if __name__ == "__main__":
    def f(x):
        return x**3 - 2*x - 5
    
    solver = BisectionMethod(verbose=True, max_iter=100, tol=1e-4)
    
    a = 1
    b = 3
    x = solver(f, a, b)
    
    
    solver.plot(0, 5)