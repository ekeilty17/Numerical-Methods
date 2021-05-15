import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import types

class Interpolator(object):

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

        self.params = self.get_parameters()

    def __call__(self, x, *args, **kwargs):
        raise NotImplementedError("__call__() function not yet implemented.")

    @staticmethod
    def _separator():
        print()
        print('*'*100)
        print()

    def get_parameters(self):
        pass

    def _get_subinterval_idx(self, x):
        if x < self.X[0] or x > self.X[-1]:
            raise ValueError(f"The interpolation is only valid between [{self.X[0]}, {self.X[-1]}].")

        for i in range(self.N-1):
            if x <= self.X[i+1]:
                return i
        
        return self.N-1
    
    def _get_subinterval(self, x):
        idx = self._get_subinterval_idx(x)
        return self.X[idx], self.X[idx+1]

    def plot(self, N=1000, save=False):

        # X values used to plot the interpolation
        a = self.X[0]
        b = self.X[-1]
        X = np.linspace(a, b, N)
        
        # plotting axes
        ax = plt.gca()
        ax.grid()
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')

        # plotting interpolation
        Y_approx = [self(x) for x in X]
        plt.plot(X, Y_approx, 'b-', label=f"{self.func_name}(x)")
        
        # plotting given points
        if not self.f is None:
            Y_actual = [self.f(x) for x in X]
            plt.plot(X, Y_actual, color='orange', alpha=0.7, label="f(x)")
        
        plt.plot(self.X, self.F, color='k', marker='o', linestyle='None', markersize=5, label="given points")

        # all the stuff to make the plot pretty
        plt.legend()
        #ax.set_ylim(-1, 2)

        title = f"{self.name.title()} Interpolation (n={self.N})"
        plt.title(title)
        
        if save:
            plt.savefig(title)
        else:
            plt.show()
        plt.clf()
    
    def _integrate_subinterval(self, Pi, a, b):
        integral = sym.integrate(Pi)
        Fb = integral.subs(self.var, b)
        Fa = integral.subs(self.var, a)
        
        if self.verbose:
            print(f"Integral of {self.func_name} from {round(a, 4)} to {round(b, 4)} = {float(Fb):.4f} - {float(Fa):.4f} = {float(Fb - Fa):.4f}")
        
        return Fb - Fa

    def integrate(self, a=None, b=None):
        
        # getting the bounds and the subinterval of each bound
        if a is None:
            a = self.X[0]
        if b is None:
            b = self.X[-1]

        a_idx = self._get_subinterval_idx(a)
        b_idx = self._get_subinterval_idx(b)

        # my hacky way to deal with both continuous and piece-wise interpolations at the same time
        P = self()
        if type(P) != list:
            P = [P] * (self.N - 1)
        
        # special case: bounds are within the same subinterval
        if a_idx == b_idx:
            return self._integrate_subinterval(P[a_idx], a, b)

        result = 0

        # Do a_idx subinterval
        result += self._integrate_subinterval(P[a_idx], a, self.X[a_idx+1])

        # do all subintervals in between
        for i in range(a_idx+1, b_idx):
            result += self._integrate_subinterval(P[i], self.X[i], self.X[i+1])
        
        # do b_idx subinterval
        result += self._integrate_subinterval(P[a_idx], self.X[b_idx], b)

        if self.verbose:
            print("-"*75)
            print(f"Total = {result}")
        return result