from explicit_solver import ExplicitSolver

class RK4Solver(ExplicitSolver):

    name = "Runge-Kutta 4"

    def init_time_marching(self, u0):
        return [ [u0] ]
    
    def time_marching(self, F, Stages, h, n):
        u_n = Stages[0][-1]

        uhat_n = u_n + (h/2) * F(u_n, h*n)
        utilde_n = u_n + (h/2) * F(uhat_n, h*(n+1/2))
        ubar_n = u_n + h * F(utilde_n, h*(n+1/2))
        u_np1 = u_n + (h/6) * (F(u_n, h*n) + 2 * (F(uhat_n, h*(n+1/2)) + F(utilde_n, h*(n+1/2))) + F(ubar_n, h*(n+1)))

        if self.verbose:
            print(f"uhat_{n}   = u_{n} + (h/2) * u'_{n} = {u_n} + {h/2} * {F(u_n, h*n)}")
            print(f"uhat_{n}   = {uhat_n}\n")

            print(f"utilde_{n} = u_{n} + (h/2) * u'hat_({n}+1/2) = {u_n} + {h/2} * {F(uhat_n, h*(n+1/2))}")
            print(f"utilde_{n} = {utilde_n}\n")

            print(f"ubar_{n}   = u_{n} + h * utilde'_({n}+1/2) = {u_n} + {h} * {F(utilde_n, h*(n+1/2))}")
            print(f"ubar_{n}   = {ubar_n}\n")

            print(f"u_{n+1}      = u_{n} + (h/6) * [ u'_{n} + 2 * (u'hat_({n}+1/2) + utilde'_({n}+1/2)) + ubar'_({n}+1) ]")
            print(f"u_{n+1}      = {u_n} + {h/6} * [ {F(u_n, h*n)} + 2 * ({F(uhat_n, h*(n+1/2))} + {F(utilde_n, h*(n+1/2))}) + {F(ubar_n, h*(n+1))} ]")
            print(f"u_{n+1}      = {u_np1}\n")
            print("\n")
            print("-"*50)
            print()

        Stages[0].append(u_np1)
        return Stages

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt

    def F(u, t):
        return -u + np.sin(t)
    
    eq = RK4Solver()
    u0 = 1
    eq.solve(F, u0, h=1/64, T=1)
    eq.plot()