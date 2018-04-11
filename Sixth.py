from Third import *
from Fifth import *
import math
import numpy as np
from numpy.polynomial import legendre, polynomial


def w(x):
    if x == 0:
        return 0

    return -x * math.log(x, math.exp(1))


def f(x):
    return math.sin(x)


def Jwf(x, k):
    if x == 0:
        return 0

    return - x ** (k + 2) * ((k + 2) * math.log(x, math.exp(1)) - 1) / (k + 2) ** 2


def gauss(a, b, N, func):
    s = 0
    xs, As = legendre.leggauss(N)
    for x, A in zip(xs, As):
        xk = x * (b - a) / 2 + (b + a) / 2
        s += A * func(xk)

    return s * (b - a) / 2


def gausst(func, a, b, N):
    s = 0
    xs, As = legendre.leggauss(N)
    for x, A in zip(xs, As):
        xm = x * (b - a) / 2 + (a + b) / 2
        s += A * func(xm)
    return s * (b - a) / 2


def gauss2(a, b, N, func, m):
    h, xs = nodes(a, b, m)
    s = 0
    for i in range(m):
        s += gauss(a + i * h, a + (i + 1) * h, N, func)

    return s


def gauss_mult(func, A, B, N, m):
    s = 0
    h = (B - A) / m
    for i in range(m):
        l, r = A + i * h, A + (i + 1) * h
        s += gausst(func, l, r, N)
    return s


def main():
    print("Пиближенное вычисление интегралов при помощи КФ НАСТ\n")
    fstr = "sin(x)"
    wstr = "-xln(x)"
    print("f(x) = %s\nw(x) = %s" % (fstr, wstr))
    print("Введите пределы интегрирования, число промежутков деления и количество узлов КФ типа Гаусса (a,b,m,N):")
    a, b, m, N = lmap(float, input().split())
    m = int(m)
    N = int(N)

    J = gauss2(a, b, N, lambda x: f(x) * w(x), m)
    #J_gauss_ref = gauss_mult(lambda x: f(x) * w(x), a, b, N, m)
    #print("J = {0}".format(J_gauss_ref), end="\n\n")
    print("J = {0}".format(J), end="\n\n")

    mu = [simpson(a, (b - a) / 10000, 10000, lambda x: w(x) * x ** k) for k in range(N * 2)]
    print("Моменты весовой функции:\n", mu, end="\n\n")

    pkoeffs = np.linalg.solve([mu[i:i + N] for i in range(N)], [-x for x in mu[-N:]])
    poly = polynomial.Polynomial(pkoeffs) + polynomial.Polynomial(polynomial.polyx) ** N
    xs = poly.roots()
    pk = [pkoeffs[-i] for i in range(len(pkoeffs))]
    pk.append(1)
    print("Ортогональный многочлен:")
    pprint(pk)
    print(end="\n\n")

    print("Корни ортогонального многочлена:\n", xs, end="\n\n")

    W = polynomial.Polynomial([1])
    for x_i in xs:
        W *= polynomial.Polynomial([-x_i, 1])
    W_prime = W.deriv()

    def getp(k):
        res = [1]
        for i, x in enumerate(xs):
            res = pmul(res, [-x, 1])
        return pmul(res, [1 / W_prime(xs[k])])

    def geta(k):
        s = 0
        for i, koef in enumerate(getp(k)):
            s += koef * mu[i]
        return s

    As = [geta(k) for k in range(len(xs))]
    As = [gauss2(a, b, N, lambda x: w(x) * W(x) / ((x - x_i) * W_prime(x_i)), m) for x_i in xs]
    print("Коэффициенты КФ:\n", As, end="\n\n")
    J_gauss = sum(A * f(x) for x, A in zip(xs, As))
    print("J = ", J_gauss)


"""
    sum = 0
    for i in range(len(xs) - 1):
        first, second = xs[i], xs[i + 1]
        mu = [J(second, k) - J(first, k) for k in range(N * 2)]
        a1 = (mu[0] * mu[3] - mu[2] * mu[1]) / (mu[1] ** 2 - mu[2] * mu[0])
        a2 = (mu[2] ** 2 - mu[3] * mu[1]) / (mu[1] ** 2 - mu[2] * mu[0])
        d = a1 ** 2 - 4 * a2
        x1 = (-a1 + math.sqrt(d)) / 2
        x2 = (-a1 - math.sqrt(d)) / 2
        print(mu)
        print(a1, a2)
        print(x1, x2)
        A1 = (mu[1] - x2 * mu[0]) / (x1 - x2)
        A2 = (mu[1] - x1 * mu[0]) / (x2 - x1)
        print(A1, A2, A1 + A2, mu[0])
        I = A1 * f(x1) + A2 * f(x2)
        sum += I
    print(sum)
"""

if __name__ == "__main__":
    main()
