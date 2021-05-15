import numpy as np
import sympy as sym

from interpolator import Interpolator

class NaturalCubicSpline(Interpolator):

    name = "Natural Cubic Spline"
    func_name = "S"

    def __init__(self, X_known, F_known, var="x", verbose=False):
        super(NaturalCubicSpline, self).__init__(X_known, F_known, var=var, verbose=verbose)

    def __call__(self, x=None):
        return self.S(x)
    
    def get_parameters(self):
        # n-1 = number of intervals
        # n = number of points

        if self.verbose:
            self._separator()

        # initialize cubic coefficents from system solution
        a = np.array(self.F)
        b = np.zeros(self.N)
        c = np.zeros(self.N)
        d = np.zeros(self.N)

        """ We have to deal with a few special cases """
        if self.N < 2:
            raise ValueError("Need at least 2 points in order to interpolate.")
        if self.N == 2:
            # the equations require n-2 to exists, 
            # so we can't do an interpolation for n < 3

            # we will just return a line connecting out two data points
            b[0] = (self.F[1] - self.F[0]) / (self.X[1] - self.X[0])
            params = list(zip(a, b, c, d))
            return params[:-1]

        # compute interval lengths
        dX = np.array([self.X[i+1] - self.X[i] for i in range(self.N-1)])
        dF = np.array([self.F[i+1] - self.F[i] for i in range(self.N-1)])

        # initialize intermediate variables
        v = np.array([ 2*(dX[i] + dX[i+1]) for i in range(self.N-2) ])
        q = dF / dX
        u = np.array([ 3*(q[i+1] - q[i]) for i in range(self.N-2) ])
        T = np.zeros((self.N-2, self.N-2))

        # compute triangular system
        if self.N == 3:
            # This is another special case because T is 1x1
            T[0, 0] = v[0]
        else:
            T[0, 0], T[0, 1] = v[0], dX[1]
            for j in range(1, self.N-3):
                T[j, j-1] = dX[j]
                T[j, j]   = v[j]
                T[j, j+1] = dX[j+1]
            T[-1, -2], T[-1, -1] = dX[-2], v[-1]

        # solve T @ z = u for z
        c = np.linalg.solve(T, u)

        if self.verbose:
            print("T = ")
            print(T)
            print()
            print("u = ")
            print(u)
            print()
            print("c = ")
            print(c)

        # adding the boundary conditions c_0 = 0 and c_(n-1) = 0
        c = np.concatenate(([0], c, [0]))

        for i in range(self.N-1):
            b[i] = dF[i]/dX[i] - (dX[i]/3) * (2*c[i] + c[i+1])
            d[i] = ( c[i+1] - c[i] ) / (3 * dX[i])

        if self.verbose:
            self._separator()
            print("a =", a)
            print("b =", b)
            print("c =", c)
            print("d =", d)

        # return spline variables
        params = list(zip(a, b, c, d))

        if self.verbose:
            self._separator()
            for i, p in enumerate(params[:-1]):
                print(f"Interval: [{self.X[i]}, {self.X[i+1]}]")
                
                print(f"   parameters:  {p}")

                ai, bi, ci, di = p
                xi = self.X[i]
                x = self.var
                poly = ai + bi * (x - xi) + ci * (x - xi)**2 + di * (x - xi)**3
                print(f"   Spline:      S{i}({self.var}) = {ai} + {bi}{x} + {ci}{x}^2 + {di}{x}^3")
                print()

        return params[:-1]

    def Si(self, i, x=None):
        if x is None:
            x = self.var
        
        ai, bi, ci, di = self.params[i]
        xi = self.X[i]
        poly = ai + bi * (x - xi) + ci * (x - xi)**2 + di * (x - xi)**3

        return poly

    def S(self, x=None):
        if x is None:
            return [self.Si(i) for i in range(self.N-1)]

        # obtaining i
        i = self._get_subinterval_idx(x)
        return self.Si(i, x)

if __name__ == "__main__":

    def f(x):
        #return 1 / 1.0 / (1 + 25 * x**2)
        return np.exp(x)
    
    
    X_known = [-1, 0, 1]
    F_known = [1, 0, 1]

    S = NaturalCubicSpline(X_known, F_known, verbose=True)
    S.plot()

    """
    #integral = sym.integrate(sym.exp(x))
    a = X_known[0]
    b = X_known[-1]
    print(a, b)
    print("actual:", np.exp(b) - np.exp(a))
    """

    """
    import matplotlib.pyplot as plt
    X = np.linspace(a, b, 100)
    Y = [integral.subs(x, x0) for x0 in X]
    plt.plot(X, Y)
    plt.show()
    """
