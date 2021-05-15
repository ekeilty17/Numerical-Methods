from interpolator import Interpolator

class LagrangePolynomials(Interpolator):

    name = "Lagrange Polynomial"
    func_name = "P"

    def __init__(self, X_known, F_known, var="x", verbose=False):
        super(LagrangePolynomials, self).__init__(X_known, F_known, var=var, verbose=verbose)

    def __call__(self, x=None):
        return self.P(x)
    
    def get_parameters(self):
        
        params = []
        for i in range(self.N):
            Li = self.L(i)
            params.append( Li )

            if self.verbose:
                print(f"L{i} = {Li}")
        
        return params

    def L(self, i, x=None):
        if x is None:
            x = self.var

        Li = 1
        for j in range(self.N):
            if j == i:
                continue
            Li *= (x - self.X[j]) / (self.X[i] - self.X[j])
        
        return Li

    def P(self, x=None):
        result = 0
        for i in range(self.N):
            Fi = self.F[i]
            Li = self.L(i, x)

            # This is why slower for some reason, I'm not sure why
            # and it's not due to the if statement, I checked that. Evaluating sympy functions is just slow
            #Li = self.params[i] if x is None else self.params[i].subs(self.var, x)
            result += Fi * Li

        return result
    


if __name__ == "__main__":
    
    import numpy as np

    def f(x):
        #return 1 / 1.0 / (1 + 25 * x**2)
        return np.exp(x)
    
    a, b = 0, 1
    n = 31
    X_known = np.linspace(a, b, n)
    #X_known = [1, 2, 3, 5, 4]

    P = LagrangePolynomials(X_known, f)
    P.plot()

    """
    P.verbose = True
    print( "approximation:", P.integrate() )

    import sympy as sym
    x = sym.Symbol('x')

    #integral = sym.integrate(sym.exp(x))
    F = P()
    integral = sym.integrate(F)
    print(a, b)
    print("Full:", integral.subs(x, b) - integral.subs(x, a))
    print("actual:", np.exp(b) - np.exp(a))
    """

    """
    def f(x):
        #return 1 / 1.0 / (1 + 25 * x**2)
        return np.exp(x)
    
    X_known = [-1, 0, 1]
    F_known = [1, 0, 1]

    L = LagrangePolynomials(X_known, F_known, verbose=True)
    L.plot()
    """