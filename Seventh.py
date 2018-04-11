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
taylor_poly = [coef / float(math.factorial(i)) for i, coef in enumerate(taylor_coeffs)]


def taylor(x):
    return pval(taylor_poly, x)


def adams_sd(x, ys, h):
    xs = [x - (4 - i) * h for i in range(5)]
    yps = [h * y_pr(x, y) for x, y in zip(xs, ys)]
    return sepdiffs(xs, yps)


def adams(xs, ys, h, N):
    for i in range(N):
        sd = adams_sd(xs[-1], ys[-5:], h)
        xs.append(xs[-1] + h)
        ys.append(ys[-1] + sd[4][0] + sd[3][1] / 2 + 5 * sd[2][2] / 12 + 3 * sd[1][3] / 8 + 251 * sd[0][4] / 720)

    return xs, ys


def runge_kutta(x_0, y_0, N, h):
    xs, ys = [x_0], [y_0]
    for _ in range(N):
        x, y = xs[-1], ys[-1]
        k1 = h * y_pr(x, y)
        k2 = h * y_pr(x + h / 2, y + k1 / 2)
        k3 = h * y_pr(x + h / 2, y + k2 / 2)
        k4 = h * y_pr(x + h, y + k3)
        xs.append(x + h)
        ys.append(y + (k1 + 2 * k2 + 2 * k3 + k4) / 6)

    return xs, ys


def euler1(x0, y0, N, h):
    xs, ys = [x0], [y0]
    for _ in range(N):
        x, y = xs[-1], ys[-1]
        xs.append(x + h)
        ys.append(y + h * y_pr(x, y))

    return xs, ys


def euler2(x0, y0, N, h):
    xs, ys = [x0], [y0]
    for _ in range(N):
        x, y = xs[-1], ys[-1]
        xs.append(x + h)
        ys.append(y + h * y_pr(x + h / 2, y + h / 2 * y_pr(x, y)))

    return xs, ys


def euler3(x0, y0, N, h):
    xs, ys = [x0], [y0]
    for _ in range(N):
        x, y = xs[-1], ys[-1]
        xs.append(x + h)
        ys.append(y + h / 2 * (y_pr(x, y) + y_pr(x + h, y + h * y_pr(x, y))))

    return xs, ys


def main():
    x0, y0 = 0, 1
    print("Дифференциальное уравнение:\ny' = -3y + y^2\nЗадача Коши\ny(0) = 1")
    print("Точное решение: y =  3/(2exp(3x) + 1)")
    print("Введите число шагов N и шаг h > 0:")
    N, h = lmap(float, input().split())
    N = int(N)

    xs = [(i - 2) * h for i in range(N + 3)]
    ys = lmap(taylor, xs)
    compare("Метод разложения в ряд Тейлора", xs, ys, sol)

    xs = [(i - 2) * h for i in range(5)]
    ys = lmap(taylor, xs)

    xs, ys = adams(xs, ys, h, N - 2)
    compare("Метод Адамса 4-го порядка", xs, ys, sol)

    xs, ys = runge_kutta(x0, y0, N, h)
    compare("Метод Рунге-Кутты 4-го порядка", xs, ys, sol)

    xs, ys = euler1(x0, y0, N, h)
    compare("Метод Эйлера", xs, ys, sol)

    xs, ys = euler2(x0, y0, N, h)
    compare("Улучшенный метод Эйлера", xs, ys, sol)

    xs, ys = euler3(x0, y0, N, h)
    compare("Улучшенный метод Эйлера 2", xs, ys, sol)


if __name__ == "__main__":
    main()
