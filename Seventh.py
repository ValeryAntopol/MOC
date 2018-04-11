import math
from tabulate import tabulate

from Third import *


def compare(header, xs, ys, sol):
    print(
        header,
        tabulate(zip(range(len(xs)), xs, ys, [abs(y - sol(x)) for x, y in zip(xs, ys)]),
                 headers=['#', 'x', 'y', 'err'], floatfmt='.8f'),
        sep='\n', end='\n\n')


def y_pr(x, y_x):
    return -3 * y_x + y_x ** 2


def sol(x):
    return 3 / (2 * math.exp(3 * x) + 1)


taylor_coeffs = [1.0, -2.0, 2.0, 6.0, -30.0, -35.0]
taylor_poly = [coef * float(math.factorial(i)) for i, coef in enumerate(taylor_coeffs)]


def taylor(x):
    return pval(taylor_poly, x)


def adams_sd(x, ys, h):
    xs = [x - (4 - i) * h for i in range(5)]
    yps = [y_pr(x, y) for x, y in zip(xs, ys)]
    return sepdiffs(xs, yps)


def adams(xs, ys, h, N):
    for i in range(N):
        sd = adams_sd(xs[-1], ys[-5:], h)
        xs.append(xs[-1] + h)
        ys.append(ys[-1] + sd[4][0] + sd[3][1] / 2 + 5 * sd[2][2] / 12 + 3 * sd[1][3] / 8 + 251 * sd[0][4] / 720)

    return xs, ys


def main():
    x0, y0 = 0, 1
    print("Дифференциальное уравнение:\ny' = -3y + y^2\nЗадача Коши\ny(0) = 1")
    print("Точное решение: y =  3/(2exp(3x) + 1)")
    print("Введите число шагов N и шаг h > 0:")
    N, h = lmap(float, input().split())
    N = int(N)
    xs = [(i - 2) * h for i in range(5)]
    ys = lmap(taylor, xs)
    compare("Метод разложения в ряд Тейлора", xs, ys, sol)
    xsa, ysa = adams(xs, ys, h, N)
    compare("Метод Адамса", xs, ys, sol)


if __name__ == "__main__":
    main()
