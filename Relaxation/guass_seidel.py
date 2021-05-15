from classical_relaxation import ClassicalRelaxation

import numpy as np

class GuassSeidelRelaxation(ClassicalRelaxation):

    def __init__(self, tol=1e-5, max_iter=100, verbose=False):
        super(GuassSeidelRelaxation, self).__init__(tol=tol, max_iter=max_iter, verbose=verbose)
        
    def __call__(self, A, f, X0=None):
        L, D, U = self.decompose(A)
        H = -(L + D)
        return super(GuassSeidelRelaxation, self).__call__(A=A, f=f, H=H, X0=X0)


if __name__ == "__main__":
    
    Relax = GuassSeidelRelaxation(verbose=True)

    A = Relax.B(3, (1, -2, 1))
    f = np.array([1, 2, 3])

    X0 = [1, 1, 1]
    Relax(A, f, X0=X0)
    print()
    print("Exact Solution:   ", Relax.exact_solution(A, f))