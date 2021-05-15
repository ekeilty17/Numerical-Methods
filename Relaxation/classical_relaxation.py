import numpy as np

class ClassicalRelaxation(object):

    def __init__(self, tol=1e-5, max_iter=100, verbose=False):
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.iterations = []
    
    def __call__(self, A, f, H, X0=None):
        A = np.array(A)
        f = np.array(f)
        self.H = np.array(H)
        m, n = A.shape

        if X0 is None:
            X0 = np.ones(n)
        
        self.iterations = []
        return self.solve(A, f, X0) 
    
    @staticmethod
    def decompose(A):
        L = np.tril(A, k=-1)
        D = np.diag(np.diag(A))
        U = np.triu(A, k=1)

        return L, D, U

    @staticmethod
    def B(M, diag, p=False):
        if len(diag) % 2 == 0:
            raise ValueError("Please use an odd number of parameters as to not be ambiguous")
        if len(diag) > M:
            raise ValueError(f"Number of parameters is greater than the size of the matrix ({M})")

        half = len(diag) // 2
        A = np.zeros((M, M)) 

        for k, d in enumerate(diag, -1 * half):
            A = A + d * np.eye(M, k=k)

        # periodic...meaning diagonal loops around
        if p and len(diag) > 1:
            # looping around on the top
            for k, d in enumerate(diag[:half], -1 * half):
                A = A + d * np.eye(M, k=M + k)
        
            # looping around on the bottom
            for k, d in enumerate(diag[half+1:], 1): 
                A = A + d * np.eye(M, k=k-M)

        return A

    @staticmethod
    def exact_solution(A, f):
        A = np.array(A)
        f = np.array(f)
        return np.linalg.inv(A) @ f

    def solve(self, A, f, X0):
        A = np.array(A)
        f = np.array(f)
        m, n = A.shape

        # initialization
        H_inv = np.linalg.inv(self.H)
        self.G = np.identity(n) + H_inv @ A
        if self.verbose:
            print("A =\n", A)
            print()
            print("f =", f)
            print()
            print("H =\n", self.H)
            print()
            print("H^(-1) =\n", H_inv)
            print()
            print("G = I + H^(-1) @ A =\n", self.G)
            print()
            print("*"*100)
            print()
        
        # more initialization
        X = np.array(X0, dtype=float).copy()
        error = A @ X - f
        self.iterations.append(X)

        k = 0
        while np.linalg.norm(error) > self.tol and k < self.max_iter:
            k += 1
            
            # iterate
            X_new = self.G @ X - H_inv @ f
            error = A @ X_new - f

            if self.verbose:
                print(f"Iteration {k}: X{k-1} = {X}")
                print(f"\tX{k} = G @ X{k-1} - H^(-1) @ f = {self.G @ X} - {H_inv @ f} = {X_new}")
                print(f"\tError = A @ X{k} - f = {A @ X_new} - {f} = {error}")
                print("\n")
            
            X = X_new
            self.iterations.append(X.copy())

        if self.verbose:
            print(f"Relaxed Solution:  X =", X)
        return X


if __name__ == "__main__":

    Relax = ClassicalRelaxation(max_iter=20, verbose=True)

    A = Relax.B(3, (1, 2, 1))
    f = np.array([1, 2, 3])

    L, D, U = ClassicalRelaxation.decompose(A)
    H = -D

    print("L =\n", L)
    print("D =\n", D)
    print("U =\n", U)
    print("\n")

    X0 = [1, 1, 1]
    Relax(A, f, H, X0=X0)
    print()
    print("Exact Solution:   ", Relax.exact_solution(A, f))