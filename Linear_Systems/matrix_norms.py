import numpy as np

def frobenius_norm(A):
    return np.linalg.norm(A)


def condition_number(A):
    return frobenius_norm(A) * frobenius_norm(np.linalg.inv(A))