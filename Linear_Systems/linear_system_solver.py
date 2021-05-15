import numpy as np

class LinearSystemSolver(object):

    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def __call__(self, A, b, *args, **kwargs):
        raise NotImplementedError("__call__() function not yet implemented.")
    
    @staticmethod
    def _display_numpy_matrix(M, precision=3):
        M_rounded = np.around(M, precision)
        M_str = ' ' + str(M_rounded)[1:-1] \
                            .replace('[', '|').replace(']', '|') \
                            .replace('. ', '  ').replace('.|', ' |') \
                            .replace(' 0 ', ' \033[90m0\033[0m ') \
                            .replace('|0 ', '|\033[90m0\033[0m ') \
                            .replace(' 0|', ' \033[90m0\033[0m|')
        return M_str.split('\n')
    
    @staticmethod
    # TODO: generalize to non-square matrices
    def _display_matrix_update(*matrices):
        matrices = list(matrices)
        m, n = matrices[0].shape
        Product = np.identity(m)
        
        MULT = [''] * int(m/2) + ['@'] + [''] * int(m/2 + 1)
        EQUAL = [''] * int(m/2) + ['='] + [''] * int(m/2 + 1)
        
        output = []
        for M in matrices:
            output.append( LinearSystemSolver._display_numpy_matrix(M) )
            output.append( MULT )
            Product = Product @ M

        output[-1] = EQUAL
        output.append( LinearSystemSolver._display_numpy_matrix(Product) )

        for args in zip(*output):
            print('\t'.join(args))
    
    @staticmethod
    def is_lower_triangular(L):
    
        for i in range(1, len(L)):
            for j in range(i+1, len(L[i])):
                if L[i][j] != 0:
                    return False
        
        return True

    @staticmethod
    def is_upper_triangular(U):
        
        for i in range(1, len(U)):
            for j in range(0, i-1):
                if U[i][j] != 0:
                    print(i, j)
                    return False
        
        return True

    # FIXME: make these work for general m x n matrices
    @staticmethod
    def forward_substitution(L, b):
        """
        solving L @ x = b for x 
        where L is a lower triangular matrix
        """
        m, n = L.shape

        x = np.zeros(n, dtype=float)
        for i in range(n):
            s = sum([L[i, j] * x[j] for j in range(0, i)])
            x[i] = (b[i] - s) / L[i, i]

        return x

    @staticmethod
    def backward_substitution(U, b):
        """
        solving U @ x = b for x 
        where U is an upper triangular matrix
        """
        m, n = U.shape

        x = np.zeros(n, dtype=float)
        for i in reversed(range(n)):
            s = sum([U[i, j] * x[j] for j in range(i+1, n)])
            x[i] = (b[i] - s) / U[i, i]
            
        return x
    
    @staticmethod
    def pseudo_inverse(A):
        """
        The Moore-Penrose Pseudo Inverse
        """
        m, n = A.shape

        if m < n:                   # under-determined system
            return A.T @ np.linalg.inv( A @ A.T )
        elif m == n:                # uniquely-determined system
            return np.linalg.inv(A)
        else:                       # overdetermined system
            return np.linalg.inv( A.T @ A ) @ A.T

    @staticmethod
    def analytical_solution(A, b):
        A_pseudo_inverse = LinearSystemSolver.pseudo_inverse(A)
        return A_pseudo_inverse @ b