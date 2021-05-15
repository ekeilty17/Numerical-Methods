from classical_relaxation import ClassicalRelaxation

import numpy as np

class SuccessiveOverRelaxation(ClassicalRelaxation):

    def __init__(self, w=None, tol=1e-5, max_iter=100, verbose=False):
        super(SuccessiveOverRelaxation, self).__init__(tol=tol, max_iter=max_iter, verbose=verbose)
        self.w = w

        # w = 2 --> Gauss-Seidel Relaxation method

    def __call__(self, A, f, X0=None):
        m, n = A.shape

        if self.w is None:
            # optimum w
            self.w = 2 / (1 + np.sin(np.pi / (m+1)))
        H = self.B(m, (-1, 2/self.w, 0))

        return super(SuccessiveOverRelaxation, self).__call__(A=A, f=f, H=H, X0=X0)


if __name__ == "__main__":
    
    Relax = SuccessiveOverRelaxation(w=1, verbose=True)

    A = Relax.B(3, (1, -2, 1))
    f = np.array([1, 2, 3])

    X0 = [1, 1, 1]
    Relax(A, f, X0=X0)
    print()
    print("Exact Solution:   ", Relax.exact_solution(A, f))