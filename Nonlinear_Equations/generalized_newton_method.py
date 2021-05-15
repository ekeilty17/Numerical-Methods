import numpy as np
import matplotlib.pyplot as plt

from nonlinear_system_solver import NonlinearSystemSolver

class GeneralizedNewtonMethod(NonlinearSystemSolver):

    name = "Generalized Newton's Method"

    def __init__(self, tol=1e-5, max_iter=100, verbose=False):
        super(GeneralizedNewtonMethod, self).__init__(tol=tol, max_iter=max_iter, verbose=verbose)

    def __call__(self, f, df, ddf, X0):
        if df is None or ddf is None:
            raise NotImplementedError("SymPy automatic differentiation not implemented yet.")

        self.f = f
        self.df = df
        self.ddf = ddf
        self.iterations = []
        return self.solve(f=f, df=df, ddf=ddf, X0=X0)

    def solve(self, f, df, ddf, X0):

        # initialization
        X = np.array(X0, dtype=float).copy()
        dX = np.ones_like(X, dtype=float)       # initialize with 1s and not 0s so it fails the self.tol check
        self.iterations.append(X)

        k = 0
        while np.linalg.norm(dX) > self.tol and k < self.max_iter:
            k += 1
            
            f_k = f(X)
            df_k = df(X)
            ddf_k = ddf(X)
            dX = np.linalg.solve(ddf_k, -df_k)

            if self.verbose:
                print(f"Iteration {k}: X{k-1} = {X}")
                print("\nddf = ")
                print(ddf_k)
                print("\n-df = ", -df_k)
                print(f"\n\tX{k} =", X + dX)
                print(f"\t|X{k-1} - X{k}| =", np.linalg.norm(dX))
                print("\n")
            
            X += dX
            self.iterations.append(X.copy())

        if self.verbose:
            print(f"f({X}) = {f(X)}")
        return X


if __name__ == "__main__":

    def f(X, alpha=10):
        x, y = X
        
        return (1 - x)**2 + alpha * (y - x**2)**2

    def df(X, alpha=10):
        x, y = X
        
        fx = 2 * (x - 1) + 2 * alpha * (x**2 - y) * 2 * x
        fy = 2 * alpha * (y - x**2)
        return np.array([ fx, fy ])

    def ddf(X, alpha=10):
        x, y = X
       
        fxx = 2 + 12 * alpha * x**2 - 4 * alpha * y
        fxy = fyx = -4 * alpha * x
        fyy = 2 * alpha
        return np.array([
                    [fxx, fxy],
                    [fyx, fyy]
                ])
    
    solver = GeneralizedNewtonMethod(verbose=True)
    X0 = [0, 1]
    X = solver(f, df, ddf, X0)
