from taylor_table import taylor_table

def tm_taylor_table(*args, K=1, depth=5, solve=True, verbose=True, backwards=False):
    '''
    args is assumed to be a tuple of lists (or tuples)
    each list represents the order of the derivative, 
    and each element in the list represents the subscript

    Ex: u_{n+1} = u_n + u_{n-1} + h * (u'_{n+1} + u'_n + u'_{n-1})
        --> ( [0, 1], [1, 0, -1] )
    '''
    
    Table, sol, error = taylor_table(0, K, *args, depth=depth, h='h', n='n', verbose=verbose, backwards=backwards)

    if verbose:
        # getting leading error term
        for e in range(len(error)):
            if error[e] != 0:
                break
        
        print(f"Method Order: {e-1}")
        print()

        print(f"Local Error:  O(h^{e})")
        print(f"Global Error: O(h^{e-1})")
        print()

    return sol, error

if __name__ == "__main__":
    tm_taylor_table([0, -1], [1], depth=10)