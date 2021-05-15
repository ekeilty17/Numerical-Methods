from collections import defaultdict
import math

def taylor_series_expansion(p, k, depth=10):
    '''
    taylor series expansion of u^(p)_{n+k}
    '''
    u = defaultdict(lambda: (0, 1))

    for i in range(depth+1):
        num = k**i
        den = math.factorial(i)
        gcd = math.gcd(num, den)
        u[i + p] = ( num//gcd, den//gcd )
    
    return u

def taylor_coefficients(coef, p, k, depth=10, simple_only=True):
    '''
    prints out row of taylor table in human-readable form for the entry:
        a, b = coef
        (a/b) * u^(p)_{n+k}
    '''

    u = taylor_series_expansion(p=p, k=k, depth=depth)
    #print(u)

    # standardizing form of coef variable
    if type(coef) == int:
        coef = (coef, )
    if len(coef) == 1:
        coef = (coef[0], 1)

    row = ['0'] * (depth + 1)
    simplified_row = ['0'] * (depth + 1)
    for i in range(depth + 1):
        num, den = u[i]

        row[i] = f"({coef[0]}/{coef[1]}) * ({num}/{den})"

        # combining row coefficient and taylor series coefficient
        num *= coef[0]
        den *= coef[1]
        gcd = math.gcd(num, den)
        num = num // gcd
        den = den // gcd
        
        if num == 0:
            continue
        
        simplified_row[i] = str(num)
        if den != 1:
            simplified_row[i] += '/' + str(den)

    s = f"({coef[0]}/{coef[1]}) * "
    if p > 0:
        s += f"(d_{'x' * p} u)"
    else:
        s += "u"
    s += "_{j"
    if k < 0:
        s += '-' + str(abs(k))
    elif k > 0:
        s += '+' + str(abs(k))
    s += '}'
    print(s)

    if not simple_only:
        print()
        print("(Row Coef) * (Taylor Coef)")
        s = '|   ' + '   |   '.join(row) + '   |'
        print('-' * len(s))
        print(s)
        print('-' * len(s))
        
        print()
        print("Simplified Value")
    
    s = '|   ' + '   |   '.join(simplified_row) + '   |'
    print('-' * len(s))
    print(s)
    print('-' * len(s))
    print()

    return simplified_row

if __name__ == "__main__":

    # 3.2
    """
    # using coef = +/- 1 as a placeholder for "no coefficient"
    taylor_coefficients( 1, 1, -1, depth=4, simple_only=True)        # (d_x u)_{j-1}
    taylor_coefficients( 1, 1,  0, depth=4, simple_only=True)        # (d_x u)_{j}
    taylor_coefficients(-1, 0, -1, depth=4, simple_only=True)        # u_{j-1}
    taylor_coefficients(-1, 0,  0, depth=4, simple_only=True)        # u_{j}
    taylor_coefficients(-1, 0,  1, depth=4, simple_only=True)        # u_{j+1}
    """

    # 3.6
    # using coef = +/- 1 as a placeholder for "no coefficient"
    taylor_coefficients( 1, 2, -1, depth=4, simple_only=True)        # (d_xx u)_{j-1}
    taylor_coefficients( 1, 2,  0, depth=4, simple_only=True)        # (d_xx u)_{j}
    taylor_coefficients( 1, 2,  1, depth=4, simple_only=True)        # (d_xx u)_{j+1}
    taylor_coefficients(-1, 0, -1, depth=4, simple_only=True)        # u_{j-1}
    taylor_coefficients(-1, 0,  0, depth=4, simple_only=True)        # u_{j}
    taylor_coefficients(-1, 0,  1, depth=4, simple_only=True)        # u_{j+1}