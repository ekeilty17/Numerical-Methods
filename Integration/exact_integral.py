
import sympy as sym

def exact_integral(f, a, b, var='x', verbose=True):

    if type(var) == str:
        var = sym.Symbol(var)

    I = sym.integrate(f(var), var)
    exact = I.subs(var, b) - I.subs(var, a)

    if verbose:
        print(f"Exact Integral: {exact} = {float(exact)}")

    return exact