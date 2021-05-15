from explicit_solver import ExplicitSolver

class McCormmackSolver(ExplicitSolver):

    name = "McCormmack Method"

    def init_time_marching(self, u0):
        return [ [u0] ]
    
    def time_marching(self, F, Stages, h, n):
        u_n = Stages[0][-1]
        
        utilde_n = u_n + h * F(u_n, h*n)
        u_np1 = (1/2) * (u_n + utilde_n + h * F(utilde_n, h*n))

        if self.verbose:
            print(f"utilde_{n} = u_{n} + h * u'_{n} = {u_n} + {h * F(utilde_n, h*n)} = {utilde_n}")
            print(f"u_{n+1} = (1/2) * (u_{n} + utilde_{n} + h * utilde'_{n})")
            print(f"u_{n+1} = (1/2) * ({u_n} + {utilde_n} + {h * F(utilde_n, h*n)})")
            print(f"u_{n+1} = (1/2) * ({u_n + utilde_n + h * F(utilde_n, h*n)})")
            print(f"u_{n+1} = {u_np1}")
            print()

        Stages[0].append(u_np1)
        return Stages


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt

    def F(u, t):
        return -u + np.sin(t)
    
    eq = McCormmackSolver()
    u0 = 1
    eq.solve(F, u0, h=1/64, T=1)
    eq.plot()