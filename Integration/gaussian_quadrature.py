import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from scipy.special.orthogonal import p_roots

from compound_quadrature import CompoundQuadrature

class GaussianQuadrature(CompoundQuadrature):

    name = "Gaussian Quadrature"

    def __init__(self, D, f, var="x", verbose=False):
        self.a, self.b = D
        self.f = f

        if self.a >= self.b:
            raise ValueError(f"a ({a}) must be strictly less than b ({b})")

        # these are temporary values just to allow for the inheritance to work
        X_known = [self.a, self.b]
        F_known = f
        super(GaussianQuadrature, self).__init__(X_known, F_known, var=var, verbose=verbose)

    @staticmethod
    def change_of_variables(a, b):
        scale = (b-a)/2
        shift = (a+b)/2
        return scale, shift

    def integrate(self, n, m=None):
        # for simplicity, I'm not going to implement subitervals
        # I'm pretty sure with Guassian Quadrature it doesn't make sense to
        # and you will get a better approximation using the full function
        """
        if m is None:
            m = self.N

        subintervals = self._get_subintervals(m)
        """

        # I couldn't figure out how to calculate the quadrature weights manually
        # so we cheat and use scipy
        [X, A] = p_roots(n)

        if self.verbose:
            print("X:", X)
            print("A:", A)
            print()
        
        # The above assume (a, b) = (-1, 1)
        # we do a change of variables to convert them to any interval
        scale, shift = self.change_of_variables(self.a, self.b)
        X = [shift + scale * x for x in X]
        A = [scale * Ai for Ai in A]

        if self.verbose:
            print("CoV X:", X)
            print("CoV A:", A)
            print()

        # I add these only so the printing looks better
        #X = [self.a] + X + [self.b]
        #A = [0] + A + [0]

        # reinitializing everything so our functions work
        super(GaussianQuadrature, self).__init__(X, self.f, var=str(self.var), verbose=self.verbose)

        # calling our generic quadrature integration approximation
        return super(GaussianQuadrature, self).integrate(A=A)


    def annotate_plot(self):
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

if __name__ == "__main__":

    def f(x):
        return np.exp(x)

    n = 5
    integrator = GaussianQuadrature((2.0, 2.5), f, verbose=True)
    integrator.integrate(n)

    #integrator.plot()