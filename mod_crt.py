# name=mod_crt.py
# Modular-CRT prototype for computing functional square root coefficients
# Uses gmpy2 and multiprocessing to compute coefficients mod many primes
# and reconstruct exact rational coefficients assuming denominators are powers of 2.
#
# Save as mod_crt.py and run:
#   pip install gmpy2
#   python3 mod_crt.py --input-file f_coeffs.txt --N 200 --workers 8

import argparse
import math
import multiprocessing as mp
import gmpy2
from gmpy2 import mpz

gmpy2.get_context().precision = 200

# ---------- prime generator ----------
def gen_primes(start=1000000007):
    p = mpz(start)
    while True:
        p = gmpy2.next_prime(p)
        yield int(p)

# ---------- naive truncated polynomial multiply ----------
def poly_mul_trunc(a, b, mod, deg):
    n = min(len(a)-1, deg)
    m = min(len(b)-1, deg)
    c = [0] * (deg+1)
    for i in range(0, n+1):
        ai = a[i]
        if ai == 0:
            continue
        for j in range(0, min(m, deg - i) + 1):
            c[i+j] = (c[i+j] + ai * b[j]) % mod
    return c

# ---------- compute b modulo p ----------
def compute_b_mod_p(a_list, N, p):
    mod = p
    inv2 = int(gmpy2.invert(2, p))
    b = [0] * (N + 1)
    b[1] = 1 % mod
    for n in range(2, N+1):
        g_trunc = b[:n]
        acc = 0
        pow_g = [0] * (n + 1)
        pow_g[:len(g_trunc)] = g_trunc[:]
        for m in range(2, n):
            coeff_n = pow_g[n] if n < len(pow_g) else 0
            acc = (acc + b[m] * coeff_n) % mod
            pow_g = poly_mul_trunc(pow_g, g_trunc, mod, n)
        a_n = a_list[n] % mod
        rhs = (a_n - acc) % mod
        b_n = (rhs * inv2) % mod
        b[n] = b_n
    return b

# ---------- CRT combine ----------
def crt_combine(residues, primes):
    x = mpz(0)
    M = mpz(1)
    for r, p in zip(residues, primes):
        p = mpz(p)
        r = mpz(r)
        if M == 1:
            x = r
            M = p
        else:
            Minv = int(gmpy2.invert(int(M % p), int(p)))
            t = (r - x) * Minv % p
            x = x + M * t
            M = M * p
    x = int(x % M)
    M = int(M)
    return x, M

# ---------- rational reconstruction ----------
def rational_reconstruct(a_mod_m, m, max_den):
    a = mpz(a_mod_m)
    m0 = mpz(m)
    if m0 == 0:
        return None
    r0, r1 = m0, a
    s0, s1 = mpz(1), mpz(0)
    t0, t1 = mpz(0), mpz(1)
    bound = mpz(max_den)
    while r1 > bound:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        s0, s1 = s1, s0 - q * s1
        t0, t1 = t1, t0 - q * t1
    num = int(t1)
    den = int(r1)
    if den <= 0:
        return None
    if (mpz(num) * m0 - mpz(den) * a) % m0 != 0:
        return None
    g = math.gcd(abs(num), den)
    num //= g
    den //= g
    return num, den

# ---------- worker ----------
def worker_job(args):
    a_list, N, p = args
    b_mod = compute_b_mod_p(a_list, N, p)
    return p, b_mod

# ---------- orchestrator ----------
def compute_b_crt(a_list, N, bits_needed, workers):
    primes = []
    residues_list = []
    prod_bits = 0.0
    prime_gen = gen_primes(1000000007)
    pool = mp.Pool(processes=workers)
    jobs = []
    while prod_bits < bits_needed:
        p = next(prime_gen)
        primes.append(p)
        jobs.append((a_list, N, p))
        if len(jobs) >= workers:
            results = pool.map(worker_job, jobs)
            for p_res, bmod in results:
                residues_list.append((p_res, bmod))
                prod_bits += math.log2(p_res)
            jobs = []
    if jobs:
        results = pool.map(worker_job, jobs)
        for p_res, bmod in results:
            residues_list.append((p_res, bmod))
            prod_bits += math.log2(p_res)
    pool.close()
    pool.join()
    primes_order = [p for (p, _) in residues_list]
    residues_by_n = [[] for _ in range(N+1)]
    for (p, bmod) in residues_list:
        for n in range(1, N+1):
            residues_by_n[n].append(int(bmod[n]))
    return primes_order, residues_by_n

# ---------- load f coeffs ----------
def load_f_coeffs_from_file(path, N):
    with open(path, 'r') as f:
        s = f.read().strip().split()
    arr = [0] * (N + 1)
    for i in range(1, min(N, len(s)) + 1):
        arr[i] = int(s[i-1])
    return arr

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="f coefficients (a1 a2 ... aN)")
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bits", type=int, default=None, help="target bits for CRT modulus; if omitted estimated from growth")
    args = parser.parse_args()

    N = args.N
    a_list = load_f_coeffs_from_file(args.input_file, N)

    alpha = 2.34
    max_bits_est = 0
    for n in range(1, N+1):
        bits = math.log2(math.factorial(n)) + n * math.log2(alpha) + 10
        if bits > max_bits_est:
            max_bits_est = bits
    K = N
    denom_bound = 1 << K
    U = 1 << int(max_bits_est + 2)
    V = denom_bound
    bits_needed = math.log2(2 * U * V) + 2
    if args.bits:
        bits_needed = args.bits

    print("N =", N, "estimated numerator bits <=", int(max_bits_est), "denom <= 2^", K)
    print("CRT target bits:", int(bits_needed), "workers:", args.workers)

    primes, residues_by_n = compute_b_crt(a_list, N, bits_needed, args.workers)
    print("collected primes:", len(primes))

    b_rational = [None] * (N + 1)
    for n in range(1, N+1):
        residues = residues_by_n[n]
        x, M = crt_combine(residues, primes)
        rr = rational_reconstruct(x, M, denom_bound)
        if rr is None:
            print(f"rational reconstruction failed for n={n} (increase bits or K)")
            b_rational[n] = None
        else:
            num, den = rr
            b_rational[n] = (num, den)
            print(f"n={n}: b = {num}/{den}")
    with open("b_coeffs_reconstructed.txt", "w") as fo:
        for n in range(1, N+1):
            rr = b_rational[n]
            if rr is None:
                fo.write(f"{n} : FAILED\n")
            else:
                fo.write(f"{n} {rr[0]} {rr[1]}\n")
    print("done. results -> b_coeffs_reconstructed.txt")

if __name__ == "__main__":
    main()
