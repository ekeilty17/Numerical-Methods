import numpy as np
import matplotlib.pyplot as plt

from lagrange_interpolation import LagrangeInterpolation

class NewtonCotes(LagrangeInterpolation):

    name = "Newton Cotes"

    def __init__(self, X_known, F_known, var="x", verbose=False):
        
        # Note: X_known is a list of the "bounds" of each subinterval
        #       so if you set m=4, we will create 4 evenly-spaced subintervals between each pair in X
        #       setting m=None will simply use raw list

        # checking for evenly spaced intervals
        if len(X_known) > 1:
            dx = X_known[1] - X_known[0]
            for i in range(len(X_known)-1):
                if X_known[i+1] - X_known[i] != dx:
                    raise ValueError("X_known must be an evenly-spaced interval.")

        super(NewtonCotes, self).__init__(X_known, F_known, var=var, verbose=verbose)

    def integrate(self, m=None):
        if m is None:
            # take raw lists X_known and F_known
            pass
        elif m == 1:
            raise NotImplementedError("m = 1 has not been implemented.")
        else:
            if self.f is None:
                raise ValueError("We require the function in order to do the interpolation")

            X_sub = [self.X[0]]
            for i in range(len(self.X)-1):
                X_sub.extend( list(np.linspace(self.X[i], self.X[i+1], m))[1:] )
            
            self.X = X_sub
            self.F = [self.f(x) for x in self.X]
            self.N = len(self.X)
        
        return super(NewtonCotes, self).integrate(m=m)

    def error(self, M4=None):
        if M4 is None:
            raise NotImplementedError("I have not implemented automatic differentiation to obtain this automatically.")

        if self.verbose:
            self._separator()

        subintervals = self._get_subintervals(self.m)

        E_total = []
        for I in subintervals:

            X_sub = [self.X[i] for i in I]

            m = len(X_sub)
            if not m in range(2, 6+1):
                raise ValueError("Only values between [2, 6] are supported for m.")

            a, b = X_sub[0], X_sub[-1]
            d = m-1 if m%2 == 0 else m
            C = [0, 0, -1/12, -1/90, -3/80, -8/945, -275/12096]

            E_sub = C[self.m] * M4 * ((b - a) / (self.m - 1))**(d+2)
            E_total.append( np.abs(E_sub) )

            if self.verbose:
                print(f"Subinterval: {X_sub}")
                print(f"Error:       {np.abs(E_sub)}")
                print()

        if self.verbose:
            print(f"Total Error Upper Bound: {np.sum(E_total)}")
        return np.sum(E_total), E_total

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
        #return np.exp(x)
        return np.sin(1 - x/10)

    X = [0, 5, 10]
    F = [f(x) for x in X]
    integrator = NewtonCotes(X, f, verbose=True)
    integrator.integrate(m=3)

    integrator.error(M4=1e-4)
    integrator.plot()