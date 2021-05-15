import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

import lagrange_polynomials as LP

def change_of_variables(a, b, x):
    return (a+b)/2 + ( (b-a)/2 )*x

def A_NC(i, X, printing=True):
    x_i = X[i]
    a = X[0]
    b = X[-1]

    x = sym.Symbol('x')
    
    L_i = 1
    for j in range(len(X)):
        if j == i:
            continue
        x_j = X[j]
        L_i *= (x - x_j) / (x_i - x_j)
    print(f"L{i}:", L_i)

    integral = sym.integrate(L_i)
    print(f"Integral L{i}:", integral)

    A_i = integral.subs(x, b) - integral.subs(x, a)
    print(f"A{i}:", A_i)
    print()

    return A_i

def Q_NC(X, F):
    result = 0
    for i in range(len(X)):
        A_i = A_NC(i, X)
        result += F[i] * A_i
    return result

def Q_NC_error(m, M, X):

    if not m in range(2, 6+1):
        raise ValueError("Only values between [2, 6] are supported for m.")

    a, b = X[0], X[-1]
    d = m-1 if m%2 == 0 else m
    C = [0, 0, -1/12, -1/90, -3/80, -8/945, -275/12096]

    error = C[m] * M * ((b - a) / (m - 1))**(d+2)
    return np.abs( error )