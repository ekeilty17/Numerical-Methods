import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import types

class CompoundQuadrature(object):

    name = "Compound Quadrature"

    # self.N = number of given points
    # self.n = number of subintervals
    # self.m = number of points in each subinterval

    def __init__(self, X_known, F_known, var="x", verbose=False):
        
        if type(X_known) != list and not isinstance(X_known, np.ndarray):
            raise TypeError("X_known must be a list of x values.")
        if list(sorted(set(X_known))) != list(X_known):
            raise ValueError("X_known must be a strictly ascending list of values.")
        self.X = X_known

        if isinstance(F_known, types.FunctionType):
            self.f = F_known
            self.F = [self.f(x) for x in X_known]
        elif type(F_known) == list or isinstance(F_known, np.ndarray):
            self.f = None
            self.F = F_known
        else:
            raise TypeError("F_known must either be a function or a list of function values.")

        if len(self.X) != len(self.F):
            raise ValueError(f"The given X and F points must be the same dimension ({len(self.X)} != {len(self.F)})")
        
        self.N = len(self.X)

        self.var = sym.Symbol(var)
        self.verbose = verbose

    @staticmethod
    def _separator():
        print()
        print('*'*100)
        print()

    def _get_subintervals(self, m=None):
        if m is None:
            m = self.N

        k = 0
        subintervals = []
        while k <= self.N - m:
            subintervals.append( list(range(k, k+m)) )
            k += (m-1)   

        if k != self.N-1:
            print(f"Warning: the last subinterval will be of length {self.N - k}, since an interval of length {self.N} cannot be evenly divided into subintervals of length {m}")
            subintervals.append( list(range(k, self.N)) )

        return subintervals

    @staticmethod
    def _integrate_subinterval(F, A):
        result = 0
        for Fi, Ai in zip(F, A):
            # regular quadrature approximation
            result += Fi * Ai
            print(f"    + {Fi} * {Ai}")
        return result

    def integrate(self, A, m=None):
        if m is None:
            m = self.N
        elif m == 1:
            raise NotImplementedError("m = 1 has not been implemented.")

        if self.verbose:
            self._separator()

        subintervals = self._get_subintervals(m)

        # if A is a flat list, convert it into a list of subintervals
        if len(A) == self.N:
            A = [ [A[i] for i in I] for I in subintervals ]

        # checking that subintervals are correct
        for A_sub in A[:-1]:
            if len(A_sub) != m:
                raise ValueError(f"Length of each subinterval must be {m}. Currently {len(A_sub)}")

        # storing values for later
        self.A = A
        self.m = m

        # computing compound quadrature integral
        result = 0
        for I, A_sub in zip(subintervals, self.A):
            F_sub = [self.F[i] for i in I]
            
            if self.verbose:
                print(f"Subinterval: {[self.X[i] for i in I]}")

            sub_integral = self._integrate_subinterval(F_sub, A_sub)
            result += sub_integral

            if self.verbose:
                print("  " + "-"*50)
                print(f"    {sub_integral}\n")
        
        if self.verbose:
            print("Total:", result)
        
        return result
    
    def annotate_plot(self):
        raise NotImplementedError("annotate_plot() function not yet implemented.")

    def plot(self, N=1000, save=False):
        # X values used to plot the interpolation
        a = self.X[0]
        b = self.X[-1]
        X = np.linspace(a, b, N)
        
        # plotting axes
        ax = plt.gca()
        ax.grid()
        #ax.axhline(y=0, color='k')
        #ax.axvline(x=0, color='k')

        # plotting approximation
        self.annotate_plot()

        # plotting given points
        if not self.f is None:
            Y_actual = [self.f(x) for x in X]
            plt.plot(X, Y_actual, color='orange', alpha=1.0, label="f(x)")

        plt.plot(self.X, self.F, color='k', marker='o', linestyle='None', markersize=5, label="given points")

        # all the stuff to make the plot pretty
        plt.legend()

        title = f"{self.name.title()} Integration with {self.N} points and {self.m} subitervals"
        plt.title(title)
        
        if save:
            plt.savefig(title)
        else:
            plt.show()
        plt.clf()

    

if __name__ == "__main__":

    import numpy as np

    def f(x):
        return np.exp(x)
    
    X = [-3, -2, -1, 0, 1, 2, 3]
    A = list(range(len(X)))

    integrator = CompoundQuadrature(X, f, verbose=True)
    integrator.integrate(A, m=8)
    integrator.plot()
