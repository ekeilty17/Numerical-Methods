import numpy as np

from linear_system_solver import LinearSystemSolver


class LUDecomposition(LinearSystemSolver):

    # TODO: implement LU decomposition for non-square matrices

    def __init__(self, verbose=False):
        super(LUDecomposition, self).__init__(verbose=verbose)
    
    def __call__(self, A, b, pivot=True, minimize_roundoff=False):
        return self.solve(A=A, b=b, pivot=pivot, minimize_roundoff=minimize_roundoff)

    """ Matrix Operation Functions """
    # TODO: generalize for any matrix A
    @staticmethod
    def row_subtraction(A, a, i, j):
        """
        a * R_i - R_j --> R_j
        """
        E = np.identity(A.shape[0])
        E_inv = np.identity(A.shape[0])
        E[i, j] = -a
        E_inv[i, j] = a
        return E, E_inv

    @staticmethod
    def row_permutation(A, i, j):
        """
        R_i <--> R_j
        """
        E = np.identity(A.shape[0])
        Ri = E[:, i].copy()
        Rj = E[:, j].copy()
        E[:, i], E[:, j] = Rj, Ri
        E_inv = E.T
        return E, E_inv
    
    def decompose(self, A, pivot=True, minimize_roundoff=False):
        m, n = A.shape
        if m != n:
            raise ValueError("A must be a square matrix.")

        U = A.copy()
        L = np.identity(n)
        P = np.identity(n)

        if self.verbose:
            print("A:")
            print( "\n".join( self._display_numpy_matrix(A) ) )

        for j in range(n-1):                # iterating over columns

            if self.verbose:
                print()
                print("*"*120)
                if j != n-1:
                    print("Column", j+1)

            # Checking for an all zero column
            col = U[j:, j]
            all_zeros = not np.any(col)
            if all_zeros:
                raise ValueError("An all-zero column appeared, which means A must be singular. Thus, the system has no solution.")
            
            # If U[j, j] == 0, then we have a 0 value in the top row and we need to swap it
            # we choose the maximum-magnitude value in the row and swap
            # if minimize_roundoff = True, we always do this as it will minimize roundoff error in the decomposition
            if minimize_roundoff or U[j, j] == 0:
                
                if not pivot:
                    raise ValueError("There is a 0 on the main diagonal, which requires a pivot in order to decompose.")
                
                p = np.argmax(np.abs(col))
                E, E_inv = self.row_permutation(U, j, j+p)

                if self.verbose:
                    print(f"\nR{j+1} <--> R{j+p+1}")
                    
                    print("\nL:")
                    self._display_matrix_update(E, L, E_inv)

                    print("\nU:")
                    self._display_matrix_update(E, U)

                    print("\nP:")
                    self._display_matrix_update(E, P)
                    print()
                
                # Construct L, U, and P
                #   We need L @ U = P @ A 
                #   Thus, we can rewrite L @ U = (P @ L' @ P_inv) @ (P @ U')
                #   where L' and U' are the L and U you would get without premutation
                U = E @ U
                L = E @ L @ E_inv
                P = E @ P

            for i in range(j+1, m):         # iterating over rows
                if self.verbose:
                    print("-"*100)
                    print("Row", i+1)

                # Obtain row operation matrices
                a = U[i, j] / U[j, j]
                E, E_inv = self.row_subtraction(U, a, i, j)

                if self.verbose: 
                    print(f"\n({round(U[i, j], 3)}/{round(U[j, j], 3)}) * R{j+1} - R{i+1} --> R{i+1}")
                    
                    print("\nL:")
                    self._display_matrix_update(L, E_inv)

                    print("\nU:")
                    self._display_matrix_update(E, U)
                    print()

                # Construct L and U
                L = L @ E_inv
                U = E @ U
            
            if self.verbose and j != n-1:
                print("-"*100)

        if self.verbose:
            print("\n")
        
        self.L = L
        self.U = U
        self.P = P
        return L, U, P

    def solve(self, A, b, pivot, minimize_roundoff):

        # obtaining L, U, and P such that L @ U = P @ A
        L, U, P = self.decompose(A, pivot=pivot,  minimize_roundoff=minimize_roundoff)

        # permuting b to match the L and U matrices
        b = P @ b

        # solving L @ y = b for y
        y = self.forward_substitution(L, b)

        # solving U @ x = y for x
        x = self.backward_substitution(U, y)

        # returning result
        return x

if __name__ == "__main__":

    A = np.array([
        [1, -2, -2, -3],
        [3, -9, 0, -9],
        [-1, 2, 4, 7],
        [-3, -6, 26, 2]
    ])

    solver = LUDecomposition(verbose=True)
    """
    L, U, P = solver.decompose(A)

    print(L)
    print()
    print(U)
    print()
    print(P)
    print()

    print(L @ U)
    print()
    print(P @ A)
    print()
    """
    
    A = np.array([
        [1, 2, 1, -1],
        [3, 2, 4, 4],
        [4, 4, 3, 4],
        [2, 0, 1, 5]
    ])

    b = np.array([5, 16, 22, 15])

    print("A = ")
    print(A)
    print("\nb =", b)
    print()

    x = solver(A, b, minimize_roundoff=True)

    print("\n\nA @ x =", A @ x)