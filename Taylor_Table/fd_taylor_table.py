from taylor_table import taylor_table

def fd_taylor_table(*args, P=1, depth=5, verbose=True, backwards=True):

    Table, sol, error = taylor_table(P, 0, *args, depth=depth, h='dx', n='j', verbose=verbose, backwards=backwards)


    if verbose:
        # getting leading error term
        for e in range(len(error)):
            if error[e] != 0:
                break
        
        print(f"Method Order: {e-P}")
        print()

    return sol, error
    
if __name__ == "__main__":
    
    # Assignment 5, Q1
    #fd_taylor_table([-2, -1, 0, 1, 2], P=1, depth=10)

    # Question 3.1
    #fd_taylor_table([-2, -1, 0, 1], P=1, depth=10)

    # Question 3.2
    #fd_taylor_table([-1, 0, 1], [-1], P=1, depth=10)

    # Question 3.4
    #fd_taylor_table([-1, 0], [-1], P=1, depth=10)

    # Question 3.5
    #fd_taylor_table([-2, -1, 0, 1, 2], P=3, depth=10)

    # Question 3.6
    #                            V  This empty list is important because it represents that there are no (d_x u) terms
    fd_taylor_table([-1, 0, 1], [], [-1, 1], P=2, depth=10, backwards=False)