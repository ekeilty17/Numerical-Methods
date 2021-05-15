import numpy as np


def optimize(J, a, b, tol=1e-5, max_iter=100, printing=False):

    r = (3 - np.sqrt(5)) / 2
    c = a + r * (b-a)
    d = a + (1 - r) * (b-a)

    print(r)

    if printing:
        print(f"Iteration {0}")
        print(f"\t[a, b] = [{a}, {b}]")
        print(f"\t[c, d] = [{c}, {d}]")
        print()

    k = 0
    while np.abs(b - a) > tol and k < max_iter:
        
        if printing:
            print(f"Iteration {k+1}")

        if J(c) <= J(d):            # x in [a, d]
            if printing:
                print(f"\tJ(c) = {J(c)} <= {J(d)} = J(d)\n")
            #a = a
            b = d
            d = c
            c = a + r * (b-a)
        else:                       # x in [c, b]
            if printing:
                print(f"\tJ(c) = {J(c)} > {J(d)} = J(d)\n")
            a = c
            #b = b
            c = d
            d = a + (1 - r) * (b-a)

        print(f"\t[a, b] = [{a}, {b}]")
        print(f"\t[c, d] = [{c}, {d}]")
        print()

        k += 1

    return a, b

if __name__ == "__main__":

    def J(x):
        return x**3 - 4*x**2 - 2*x - 5

    a, b = optimize(J, 0, 5, printing=True)
    print(f"[a, b] = [{a}, {b}]")