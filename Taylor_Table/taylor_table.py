from taylor_series_expansion import taylor_series_expansion

from prettytable import PrettyTable
import numpy as np

def get_subscript(k):
    if k < 0:
        return f"m{abs(k)}"
    elif k == 0:
        return ""
    else:
        return f"p{abs(k)}"

def get_derivative(p):
    if p == 0:
        return ""
    else:
        return f"^({p})"
    

def taylor_table(P, K, *args, depth=5, n='n', h='h', verbose=True, backwards=False, precision=6):
    '''
    args is assumed to be a tuple of lists (or tuples)
    each list represents the order of the derivative, 
    and each element in the list represents the subscript

    base_term is the term that does not have a coefficient in front

    depth determines how many columns of the taylor table are used
    '''
    
    if verbose:

        ALPHABET = "abcdefgpqrst"
        cnt = 0

        print()
        out = f"u{get_derivative(P)}_{n}{get_subscript(K)} = ( "
        for p, Derivative_Order in enumerate(args, 0):
            
            out += f"{'' if p == 0 else f'{h}^{p} * ( '}"
            for k in Derivative_Order:
                out += f"{ALPHABET[cnt]} u{get_derivative(p)}_{n}{get_subscript(k)} + "
                cnt += 1
            
            out = out[:-2]
            out += ") + "
            
        out = out[:-2]
        print(out)
        print()

    # base_term is the term that does not have a coefficient in front
    base_term = taylor_series_expansion(p=P, k=K, depth=depth*2)

    # initializing table
    Table = PrettyTable()

    # initializing solution array
    A = []
    b = []

    # getting column names
    field_names = [""] + [f"{h}^{p} u{get_derivative(p)}_{n}" for p in range(depth+1)]
    Table.field_names = field_names

    # Adding base_term row
    row = []
    for p in range(depth+1):
        num, den = base_term[p]
        
        # add to solution variables
        b.append( num/den )
        
        # add to table variables
        entry = num if den == 1 else f"{num}/{den}" 
        row.append( entry )

    Table.add_row([f"{'' if P == 0 else f'{h}^{P} '}u{get_derivative(P)}_{n}{get_subscript(K)}"] + row)
    
    # adding blank row
    Table.add_row([''] * (depth + 2))


    L = enumerate(args, 0)
    if backwards:
        L = reversed(list(L))

    for p, pth_derivative_terms in L:
        for k in pth_derivative_terms:
            u = taylor_series_expansion(p, k, depth=depth*2)
            
            A_row = []
            row = []
            for i in range(depth+1):
                num, den = u[i]
                
                # add to solution variables
                A_row.append(num / den)

                # add to table variables
                num = -num          # make it negative because we assume everything equals 0 and only base_term is positive
                entry = num if den == 1 else f"{num}/{den}" 
                row.append( entry )
            
            A.append( A_row )
            
            Table.add_row([f"- {'' if p == 0 else f'{h}^{p} '}u{get_derivative(p)}_{n}{get_subscript(k)}"] + row)
    
    # getting resulting system
    A = np.array(A).T
    b = np.array(b)
    
    # iteratively truncating A and b until the system is uniquely determined
    sol = "Could not be solved."
    for k in reversed(range(A.shape[0])):
        try:
            sol = np.linalg.solve(A[:k, :], b[:k])
            break
        except:
            continue

    # adding blank row
    Table.add_row([''] * (depth + 2))

    # getting error
    error = np.round(A @ sol - b, precision)
    Table.add_row(["error"] + list(error))

    if verbose:
        print(Table)
        print()

    if verbose:
        print("A:")
        print(A[:k, :])
        print()
        print("b:", b[:k])
        print()
        print("sol: ", list(np.round(sol, precision)))
        print()
        print("error:", list(error))
        print()

    # printing leading error term and method order
    if verbose:
        # getting leading error term
        for e in range(len(error)):
            if error[e] != 0:
                break
        
        print(f"Truncated/Leading Error Term: {-error[e]} * {h}^{e} u{get_derivative(e)}_{n}")

    return Table, sol, error

if __name__ == "__main__":
    
    taylor_table(0, 1, [-1, 0], [0, -1], depth=5, h='h', n='n')

    taylor_table(1, 0, [-2, -1, 0, 1, 2], depth=5, h='dx', n='j')