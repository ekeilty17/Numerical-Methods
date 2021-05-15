import numpy as np
import matplotlib.pyplot as plt

class NonlinearScalarSystemSolver(object):

    colors = ['r', 'orange', 'y', 'g', 'c', 'm']

    def __init__(self, tol=1e-5, max_iter=100, verbose=False):
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.iterations = []
    
    def __call__(self, f, *args, **kwargs):
        raise NotImplementedError("__call__() function not yet implemented.")
    
    def annotate_plot(self):
        raise NotImplementedError("annotate_plot() function not yet implemented.")

    def plot(self, a, b, N=1000, save=False):

        # plotting axes
        ax = plt.gca()
        ax.grid()
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        
        # plotting function
        X = np.linspace(a, b, N)
        F = [self.f(x) for x in X]
        plt.plot(X, F, 'b-', label=f"f(x)")

        # plotting convergence annotations
        self.annotate_plot()

        # making plot look nice
        ax.set_xlim(a, b)
        plt.legend()

        title = f"{self.name.title()}"
        plt.title(title)
        
        if save:
            plt.savefig(title)
        else:
            plt.show()
        plt.clf()