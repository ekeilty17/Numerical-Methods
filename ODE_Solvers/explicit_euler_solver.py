from explicit_solver import ExplicitSolver

class ExplicitEulerSolver(ExplicitSolver):

    name = "Explicit Euler"

    def init_time_marching(self, u0):
        return [ [u0] ]
    
    def time_marching(self, F, Stages, h, n):
        u_n = Stages[0][-1]
        
        u_np1 = u_n + h * F(u_n, h*n)

        if self.verbose:
            print(f"u_{n+1} = u_{n} + h * u'_{n}")
            print(f"u_{n+1} = {u_n} + {h} * {F(u_n, h*n)}")
            print(f"u_{n+1} = {u_np1}")
            print()

        Stages[0].append(u_np1)
        return Stages


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt

    def F(u, t):
        return -u + np.sin(t)
    
    eq = ExplicitEulerSolver()
    u0 = 1
    eq.solve(F, u0, h=1/64, T=1)
    eq.plot()