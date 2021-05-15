from implicit_solver import ImplicitSolver

class ImplicitEulerSolver(ImplicitSolver):

    name = "Implicit Euler"

    def init_time_marching(self, u0):
        return [ [u0] ]
    
    def time_marching(self, F, Stages, h, n):
        u_n = Stages[0][-1]
        u = self.var
        
        RHS = u_n + h * F(u, h*(n+1))

        Stages[0].append(u_np1)
        return Stages


import numpy as np
import matplotlib.pyplot as plt

def representative_equation_implicit_euler(u0, L, mu, a, h=None, N=None, T=None, verbose=True, plot=False):
    # derivative function F = F(u, t)
    if  ((h is None) and (N is None)) or \
        ((N is None) and (T is None)) or \
        ((T is None) and (h is None)):
        raise ValueError("Must specify 2 out of 3 of the following: time step (h), total number of steps (N), or final time (T)")
    
    if h is None:
        h = T / N
    if N is None:
        N = T / h
        if int(N) != N:
            print("Warning: h does not evenly divide T, thus the endpoint will be slightly off")
        N = int(round(N, 0))
    if T is None:
        T = h * N

    U = [u0]
    for n in range(N):
        u_n = U[-1]
        u_np1 = ( u_n + h * a * np.exp(mu * (n+1) * h) ) / (1 - L * h)

        if verbose:
            print(f"u_{n+1} = ( u_{n} + h * a * e^(mu * ({n}+1) * h) ) / (1 - L * h)")
            print(f"u_{n+1} = ( {u_n} + {h * a} * e^({mu * (n+1) * h}) ) / (1 - {L * h})")
            print(f"u_{n+1} = {u_n + h * a * np.exp(mu * (n+1) * h)} / {1 - L * h}")
            print(f"u_{n+1} = {u_np1}")
            print()
        
        U.append( u_np1 )
    
    TimeSteps = np.linspace(0, T, N+1)

    if plot:
        plt.plot(TimeSteps, U)
        plt.grid()
        plt.xlabel("t")
        plt.ylabel("u")
        plt.title("Implicit Euler - Representative Equation")
        plt.show()

    return U, TimeSteps

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt

    def F(u, t):
        return -u + np.sin(t)
    
    """
    eq = ImplicitEulerSolver()
    u0 = 1
    eq.solve(F, u0, h=1/64, T=1)
    eq.plot()
    """

    representative_equation_implicit_euler(1, -1, 0, 4, h=0.1, N=100, T=None, verbose=True, plot=True)