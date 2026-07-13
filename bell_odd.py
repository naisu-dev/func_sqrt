from sympy import bell, factorial, sqrt, simplify

def conv(P, Q, N):
    R = [0]*(N + 1)
    for i, a in enumerate(P):
        if a == 0:
            continue
        for j, b in enumerate(Q):
            if i + j > N:
                break
            R[i + j] += a*b
    return R

def coeff(n):
    F = [0] + [bell(2*k - 1)/factorial(k) for k in range(1, n + 1)]

    G = [0]*(n + 1)
    G[1] = sqrt(F[1])

    for m in range(2, n + 1):
        s = 0
        P = G[:]
        for k in range(2, m):
            P = conv(P, G, n)
            s += G[k]*P[m]
        G[m] = simplify((F[m] - s)/(G[1] + G[1]**m))

    return simplify(G[n]*factorial(n))

# 分子
def A396003(n):
    return coeff(n).as_numer_denom()[0]

# 分母のlog_2
def A395978(n):
    return coeff(n).as_numer_denom()[1].as_base_exp()[1]
