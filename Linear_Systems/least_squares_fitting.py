import numpy as np

from linear_system_solver import LinearSystemSolver

class LeastSquaresFitting(LinearSystemSolver):

    def __init__(self, verbose=False):
        super(LeastSquaresFitting, self).__init__(verbose=verbose)
    
    def __call__(self, A, b):
        return self.solve(A=A, b=b)

    @staticmethod
    def rotation_matrix(A, i, j, xi, xj):

        cos_theta = xi / np.sqrt(xi**2 + xj**2)
        sin_theta = xj / np.sqrt(xi**2 + xj**2)

        G = np.identity(A.shape[0])
        G[i][i] =  cos_theta
        G[i][j] =  sin_theta
        G[j][i] = -sin_theta
        G[j][j] =  cos_theta

        return G, G.T

    def decompose(self, A):
        m, n = A.shape

        R = A.copy()
        Q = np.identity(m)
        for c in range(n):              # iterating over columns

            if self.verbose:
                print()
                print("*"*120)
                if c != m-1:
                    print("Column", c+1)

            # Checking for an all zero column
            col = R[c:, c]
            all_zeros = not np.any(col)
            if all_zeros:
                raise ValueError("An all-zero column appeared, which means A must be singular. Thus, the system has no solution.")
            
            # checking if the top row contains a zero in the first element
            if R[c, c] == 0:
                raise NotImplementedError("Row requires a pivot, which is not implemented. Use the LUP Decomposition Function")

            for i in reversed(range(c, m-1)):     # iterating over rows
                
                # getting rows to use as axis of rotate
                j = i+1
                xi = R[i, c]
                xj = R[j, c]

                # Obtain rotation matrices
                G, G_inv = self.rotation_matrix(R, i, j, xi, xj)

                if self.verbose:
                    print(f"\napply G({i+1}, {j+1}, theta)")
                    
                    print("\nQ:")
                    self._display_matrix_update(Q, G_inv)

                    print("\nR:")
                    self._display_matrix_update(G, R)
                    print()

                # Construct U and L
                R = G @ R
                Q = Q @ G_inv

            if self.verbose and j != n-1:
                print("-"*100)

        if self.verbose:
            print("\n")
        
        self.Q = Q
        self.R = R
        return Q, R


    def solve(self, A, b):

        # preform QR decomposition
        Q, R = self.decompose(A)

        # do backwards substitution ignoring the 0 entries of R
        b = Q.T @ b
        x = self.backward_substitution(R, b)

        return x

if __name__ == "__main__":

    A = np.array([
        [1, -1,  4],
        [1,  4, -2],
        [1,  4,  2],
        [1, -1,  0]
    ])

    b = [1, 2, 3, 0]

    solver = LeastSquaresFitting(verbose=True)
    
    Q, R = solver.decompose(A)

    print(np.around(Q, 10))
    print()
    print(np.around(R, 10))
    print()
    print(np.around(Q @ R, 10))
    print()

    x = solver(A, b)
    print(x)
    print(solver.analytical_solution(A, b))
    print()
    print(np.round(Q @ R @ x, 10))
    print(b)