import numpy as np
import matplotlib.pyplot as plt

from newton_cotes import NewtonCotes

class TrapezoidalRule(NewtonCotes):

    name = "Trapezoidal Rule"

    def __init__(self, X_known, F_known, var="x", verbose=False):
        super(TrapezoidalRule, self).__init__(X_known, F_known, var=var, verbose=verbose)

    def integrate(self):
        return super(TrapezoidalRule, self).integrate(m=2)

    def annotate_plot(self):

        subintervals = self._get_subintervals(self.m)
        
        plt.vlines(self.X[0], 0, self.F[0], color='black', zorder=10)

        for i in range(self.N - 1):
            a, b =   self.X[i], self.X[i+1]
            fa, fb = self.F[i], self.F[i+1]
            slope = (fb - fa) / (b - a)
                
            # making outline of bar
            xx = np.linspace(a, b, 1000)
            yy = [fa + slope * (x - a) for x in xx]
            
            plt.plot(xx, yy, color='grey')
            plt.fill_between(xx, yy, color="grey", alpha=0.3)
            plt.vlines(a, 0, fa, color='black')
            plt.vlines(b, 0, fb, color='black')

if __name__ == "__main__":
    import numpy as np

    def f(x):
        return np.exp(x)

    X = [-1, 0, 1]
    F = [f(x) for x in X]
    integrator = TrapezoidalRule(X, f, verbose=True)
    integrator.integrate()

    integrator.plot()