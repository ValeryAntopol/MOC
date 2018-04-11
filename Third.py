import numpy as np
import texttable as tt
import math


def pfit(p):
    last = len(p)
    for i in range(1, len(p) + 1):
        if p[-i] != 0:
            break

        last = len(p) - i

    return [p[i] for i in range(last + 1)]


def pmul(p1, p2):
    res = [0 for _ in range(len(p1) + len(p2))]
    for i in range(len(p1)):
        for j in range(len(p2)):
            res[i + j] += p1[i] * p2[j]
    return pfit(res)


def padd(p1, p2):
    l1 = len(p1)
    l2 = len(p2)
    lx = max(l1, l2)
    if l2 == lx:
        p1, p2 = p2, p1
        l1, l2 = l2, l1
    res = [p1[i] for i in range(lx)]

    for i in range(l2):
        res[i] += p2[i]

    return res


def pval(p, x):
    t = 1
    v = 0
    for i in range(len(p)):
        v += t * p[i]
        t *= x

    return v


def pprint(p):
    s = str(p[0])
    for i in range(1, len(p)):
        s += " + " + str(p[i]) + ("x^%d" % i)

    print(s)


def nodes(a, b, m):
    h = (b - a) / m
    xs = [a + i * h for i in range(m + 1)]
    return h, xs


def sepdiffs(xs, ys):
    sd = [[0 for _ in range(len(xs))] for _ in range(len(xs))]
    for i in range(len(xs)):
        sd[i][0] = ys[i]

    for k in range(1, len(xs)):
        for i in range(0, len(xs) - k):
            sd[i][k] = sd[i + 1][k - 1] - sd[i][k - 1]

    return sd


def newton_fw(sd, degree, _):
    res = [0]
    n = [1]
    res = padd(res, pmul(n, [sd[0][0]]))
    for k in range(1, degree + 1):
        n = pmul(pmul(n, [-k + 1, 1]), [1.0 / float(k)])
        res = padd(res, pmul(n, [sd[0][k]]))

    return res


def newton_bw(sd, degree, _):
    res = [0]
    n = [1]
    res = padd(res, pmul(n, [sd[-1][0]]))
    for k in range(1, degree + 1):
        n = pmul(pmul(n, [k - 1, 1]), [1.0 / float(k)])
        res = padd(res, pmul(n, [sd[-k - 1][k]]))

    return res


def newton_gauss_middle(sd, degree, middle):
    res = [0]
    n = [1]
    res = padd(res, pmul(n, [sd[middle][0]]))

    for k in range(1, degree + 1):
        sign = 1 - 2 * ((k + 1) % 2)
        free = (k // 2)
        n = pmul(pmul(n, [free * sign, 1]), [1.0 / float(k)])
        res = padd(res, pmul(n, [sd[middle - free][k]]))

    return res


def lmap(f, arr):
    return list(map(f, arr))


def print_table(a, b, h, xs, ys):
    tab = tt.Texttable()
    tab.set_precision(10)
    tab.header(["param", "value"])
    tab.set_deco(tt.Texttable.HEADER)
    names = ["m", "a", "b", "h"]
    tab.set_cols_width([15, 15])
    values = [len(xs) - 1, a, b, h, ]
    for row in zip(names, values):
        tab.add_row(row)

    print(tab.draw())
    print("\n")
    tab.reset()
    tab.set_cols_dtype(["f", "f"])
    tab.set_precision(10)
    tab.set_deco(tt.Texttable.HEADER)
    tab.set_cols_width([15, 15])
    tab.header(["x", "f(x)"])
    for row in zip(xs, ys):
        tab.add_row(row)

    s = tab.draw()
    print(s)


def in_range(x, rg):
    return rg[0] <= x <= rg[1]


def f(x):
    return math.sin(x) + x * x


def main():
    print("f(x) = sin(x) + x^2")
    print("Введите a, b, m:")
    a, b, m = map(float, input().split())
    m = int(m)

    h, xs = nodes(a, b, m)
    ys = lmap(f, xs)

    print_table(a, b, h, xs, ys)
    n = 0
    while True:
        print("Введите степень многочлена:")
        n = int(input())
        if 0 <= n <= m:
            break

        print("Степень многочлена должена быть целым числом не меньше 0 и не больше %d" % m)

    fw = (a, a + h)
    bw = (b - h, b)
    md = (a + (n + 1) // 2 * h, b - (n + 1) // 2 * h)

    sd = sepdiffs(xs, ys)
    tab = tt.Texttable()
    tab.set_precision(3)
    header = ["dy_%d" % i for i in range(m + 1)]
    tab.header(header)
    for i in range(m + 1):
        row = [sd[i][k] for k in range(m + 1)]
        tab.add_row(row)

    print("Таблица разделенных разностей")
    print(tab.draw())
    while True:
        middle = 0
        while True:
            print("Введите значение из промежутка [%f, %f], [%f, %f] или [%f, %f]:" % (
                fw[0], fw[1], bw[0], bw[1], md[0], md[1]))
            x = float(input())
            if in_range(x, fw):
                inter_func = newton_fw
                t = (x - a) / h
                print("Интерполяция методом Ньютона для начала таблицы")
                break
            elif in_range(x, bw):
                inter_func = newton_bw
                t = (x - b) / h
                print("Интерполяция методом Ньютона для конца таблицы")
                break
            elif in_range(x, md):
                inter_func = newton_gauss_middle
                dmin = 1000000000000
                for i in range(n // 2, len(xs) - n // 2):
                    if abs(xs[i] - x) < dmin:
                        middle = i
                        dmin = abs(xs[i] - x)

                t = (x - xs[middle]) / h
                print("Интерполяция методом Ньютона-Гаусса для середины таблицы")
                break
            else:
                print("Значение не попадает ни в один из промежутков")

        p = inter_func(sd, n, middle)
        px = pval(p, t)
        fx = f(x)

        print("P(x) = %.10f" % px)
        print("f(x) = %.10f" % fx)
        print("Absolute error: %.10f" % (math.fabs(fx - px)))

if __name__ == "__main__":
    main()
