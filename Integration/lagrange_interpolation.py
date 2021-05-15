import sympy as sym
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from compound_quadrature import CompoundQuadrature

class LagrangeInterpolation(CompoundQuadrature):

    name = "Lagrange Interpolation"

    def __init__(self, X_known, F_known, var="x", verbose=False):
        super(LagrangeInterpolation, self).__init__(X_known, F_known, var=var, verbose=verbose)

    def A_LI(self, I, i):
        a, b = self.X[I[0]], self.X[I[-1]]
        x = self.var
        
        L_i = 1
        for j in I:
            if j == i:
                continue
            L_i *= (x - self.X[j]) / (self.X[i] - self.X[j])

        integral = sym.integrate(L_i)

        A_i = integral.subs(x, b) - integral.subs(x, a)

        if self.verbose:
            print(f"Calculating Lagrangian from {a} to {b} excuding x{i}")
            print(f"  L{str(i):4s}    = ", L_i)
            print(f"  Integral = ", integral)
            print(f"  A{str(i):4s}    = ", A_i)
            print()

        return float(A_i)
    
    def integrate(self, m=None):
        if m is None:
            m = self.N
        elif m == 1:
            raise NotImplementedError("m = 1 has not been implemented.")
        
        if self.verbose:
            self._separator()
        
        subintervals = self._get_subintervals(m)
        A = [ [self.A_LI(I, i) for i in I] for I in subintervals ]
        
        if self.verbose:
            print(f"A = {A}")

        return super(LagrangeInterpolation, self).integrate(A=A, m=m)

    def annotate_plot(self):
        pass
        """
        subintervals = self._get_subintervals(self.m)
        
        plt.vlines(self.X[0], 0, self.F[0], color='black', zorder=10)

        for I, A_sub in zip(subintervals, self.A):
            g = self.X[I[0]]
            for i, Ai in zip(I, A_sub):
                
                # making outline of bar
                plt.hlines(self.F[i], g, g+Ai, color='grey')
                plt.vlines(g, 0, self.F[i], color='grey')
                plt.vlines(g+Ai, 0, self.F[i], color='grey')
                
                # filling bar
                xx = np.arange(g, g+Ai, 0.01)
                yy = [self.F[i]]*xx.shape[0]
                plt.fill_between(xx, yy, color="grey", alpha=0.3)

                g += Ai
            
            plt.vlines(g, 0, self.F[i], color='black', zorder=10)
        """


if __name__ == "__main__":

    import numpy as np

    def f(x):
        return np.exp(x)
    
    X = [-1, -0.4, 0, 0.2, 1]
    F = [f(x) for x in X]

    integrator = LagrangeInterpolation(X, f, verbose=True)
    integrator.integrate(m=1)

    integrator.plot()