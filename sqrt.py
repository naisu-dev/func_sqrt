"""
関数的平方根 近似計算ツール
-----------------------------------
g(g(x)) = f(x) を満たす g(x) のテイラー係数(=n階導関数の値)を、
f の 0 における n階導関数を入力することで求める。

sympy による厳密計算(有理数・根号)を使っているので、
本来 0 になるはずの項が浮動小数点誤差で汚れることがない。

前提: f(0) = 0 (x=0 が不動点であること)。
      f'(0) = 0 の場合は一意に決まらないため非対応。
"""

import sympy as sp
from sympy import Rational, sqrt, simplify, factorial, nsimplify


def parse_number(s):
    """'1', '-1/2', '0.5' などを sympy の厳密な数値(Rational/sqrt含む式)として受け付ける"""
    s = s.strip()
    return nsimplify(s, rational=True)


def taylor_coeffs_from_derivs(derivs):
    """f^(0)(0)..f^(N)(0) -> a_0..a_N (テイラー係数)"""
    return [d / factorial(n) for n, d in enumerate(derivs)]


def convolve_trunc(P, Q, N):
    R = [sp.Integer(0)] * (N + 1)
    for i, pi in enumerate(P):
        if pi == 0 or i > N:
            continue
        for j, qj in enumerate(Q):
            if i + j > N:
                break
            R[i + j] += pi * qj
    return R


def functional_sqrt(a, root_sign=1):
    """
    a: f のテイラー係数 a[0..N] (a[0] は 0 であること)
    戻り値: g のテイラー係数 b[0..N]  (g(x) = sum b_n x^n, g(g(x)) = f(x))
    """
    N = len(a) - 1
    if simplify(a[0]) != 0:
        raise ValueError("f(0) が 0 ではありません。x=0 を不動点とする関数のみ対応しています。")
    if N >= 1 and simplify(a[1]) == 0:
        raise ValueError("f'(0) = 0 の場合はこの方法では一意に決まりません。")

    b = [sp.Integer(0)] * (N + 1)
    b1 = sqrt(a[1]) if root_sign == 1 else -sqrt(a[1])
    b[1] = b1

    for n in range(2, N + 1):
        known_sum = sp.Integer(0)
        Gk = list(b)  # G^1
        for k in range(2, n):
            Gk = convolve_trunc(Gk, b, N)
            known_sum += b[k] * Gk[n]

        denom = simplify(b1 + b1 ** n)
        if denom == 0:
            raise ZeroDivisionError(f"n={n}: b1+b1^n が 0 になり、一意に決まりません。")
        b[n] = simplify((a[n] - known_sum) / denom)

    return b


def format_series(b, var="x"):
    terms = []
    for n in range(1, len(b)):
        c = b[n]
        if c == 0:
            continue
        term = f"{var}^{n}" if n > 1 else var
        if c == 1:
            terms.append(term)
        elif c == -1:
            terms.append(f"-{term}")
        else:
            cs = str(c)
            terms.append(f"({cs}){term}")
    return " + ".join(terms).replace("+ -", "- ")


def main():
    print("=== 関数的平方根 近似計算ツール (厳密計算版) ===")
    print("g(g(x)) = f(x) となる g(x) を、f の 0 における導関数から求めます。")
    print("(前提: f(0)=0, f'(0)!=0)\n")

    N = int(input("最大次数 N を入力してください (0階〜N階の導関数を使います): "))

    derivs = []
    print(f"\nf^(0)(0) から f^({N})(0) までを順に入力してください。")
    print("整数・小数・分数(例: -1/2)のいずれでもOKです。\n")
    for n in range(N + 1):
        while True:
            raw = input(f"  f^({n})(0) = ")
            try:
                derivs.append(parse_number(raw))
                break
            except Exception:
                print("    入力形式が正しくありません。もう一度入力してください。")

    a = taylor_coeffs_from_derivs(derivs)

    try:
        b = functional_sqrt(a)
    except (ValueError, ZeroDivisionError) as e:
        print(f"\nエラー: {e}")
        return

    print("\n--- 結果 ---")
    print("g(x) のテイラー係数・導関数の値:")
    for n in range(1, N + 1):
        bn = b[n]
        gn = simplify(b[n] * factorial(n))
        print(f"  b_{n} = {str(bn):>20}   g^({n})(0) = {gn}")

    print("\ng(x) の近似式:")
    print("  g(x) ~ " + format_series(b))


if __name__ == "__main__":
    main()
