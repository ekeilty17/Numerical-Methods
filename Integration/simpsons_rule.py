
from newton_cotes import NewtonCotes

class SimpsonsRule(NewtonCotes):

    name = "Simpson's Rule"

    def __init__(self, X_known, F_known, var="x", verbose=False):
        super(SimpsonsRule, self).__init__(X_known, F_known, var=var, verbose=verbose)

    def integrate(self):
        return super(SimpsonsRule, self).integrate(m=3)

"""
class SimpsonsRule(object):
    
    def __init__(self, X, f, verbose=False):
        self.X = X
        self.f = f
        self.verbose = verbose
    
    def integrate_subinterval(self, X_sub):
        a = X_sub[0]
        b = X_sub[-1]
        return (b - a)/6 * ( self.f(a) + 4 * self.f((a + b)/2) + self.f(b) )

    def integrate(self, m=None):
        if m is None:
            m = len(self.X) - 1
        
        if (len(self.X) - 1) % m != 0:
            raise ValueError(f"This interval (length {len(self.X)}) cannot be split evenly into {m} subintervals")

        subintervals = []
        dm = (len(self.X) - 1) // m
        for i in range(0, len(self.X)-1, dm):
            subintervals.append( self.X[i:i+dm+1] )

        result = 0
        for X_sub in subintervals:
            sub_integral = self.integrate_subinterval(X_sub)
            result += sub_integral

            if self.verbose:
                print(f"{X_sub} --> {sub_integral}")
        
        if self.verbose:
            print(result)
        return result
"""

if __name__ == "__main__":
    import numpy as np

    def f(x):
        #return np.exp(x)
        return np.sin(1 - x/10)

    X = [0, 5, 10]
    F = [f(x) for x in X]
    integrator = SimpsonsRule(X, f, verbose=True)
    integrator.integrate()

    integrator.plot()