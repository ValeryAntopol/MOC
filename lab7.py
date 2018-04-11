import math

from tabulate import tabulate

##############################################################################
# Параметры задачи
##############################################################################
def y_prime(x, y_x):
    "y' = -3y + y^2"
    #"y'(x) = -y(x) * (2 - cos(x))"
    #return -y_x * (2 - math.cos(x))
    return -3 * y_x + y_x ** 2

# Задача Коши
x_0, y_0 = 0.0, 1.0

# Решение з.К. y(0) = 1
def sol(x):
    "y =  3/(2exp(3x) + 1)"
    #"y(x) = e^(sin(x) - 2x)"
    #return math.exp(math.sin(x) - 2 * x)
    return 3 / (2 * math.exp(3 * x) + 1)

#taylor_coeffs = [1.0, -1.0, 1.0, -2.0, 5.0, -11.0]
taylor_coeffs = [1.0, -2.0, 2.0, 6.0, -30.0, -35.0]
##############################################################################


def taylor_eval(coeffs, x, x_0):
    res = 0
    mul = 1
    for i, coeff in enumerate(coeffs):
        res += coeffs[i] * mul
        mul *= (x - x_0) / (i + 1)
    return res


def taylor(coeffs, x_0, N, h):
    xs = [x_0 + i * h for i in range(N)]
    ys = [taylor_eval(coeffs, x, 0) for x in xs]
    return xs, ys


def euler1(y_prime, x_0, y_0, N, h):
    xs, ys = [x_0], [y_0]

    for _ in range(N):
        x, y = xs[-1], ys[-1]
        xs.append(x + h)
        ys.append(y + h * y_prime(x, y))

    return xs, ys


def euler2(y_prime, x_0, y_0, N, h):
    xs, ys = [x_0], [y_0]

    for _ in range(N):
        x, y = xs[-1], ys[-1]
        xs.append(x + h)
        ys.append(y + h * (y_prime(x, y) + y_prime(x + h, y + h * y_prime(x, y))) / 2)

    return xs, ys


def euler3(y_prime, x_0, y_0, N, h):
    xs, ys = [x_0], [y_0]

    for _ in range(N):
        x, y = xs[-1], ys[-1]
        xs.append(x + h)
        ys.append(y + h * y_prime(x + h / 2, y + h * y_prime(x, y) / 2))

    return xs, ys


def runge_kutta(y_prime, x_0, y_0, N, h):
    xs, ys = [x_0], [y_0]

    for _ in range(N):
        x, y = xs[-1], ys[-1]

        k1 = h * y_prime(x, y)
        k2 = h * y_prime(x + h / 2, y + k1 / 2)
        k3 = h * y_prime(x + h / 2, y + k2 / 2)
        k4 = h * y_prime(x + h, y + k3);

        xs.append(x + h)
        ys.append(y + (k1 + 2 * k2 + 2 * k3 + k4) / 6)

    return xs, ys


def adams_fd(y_prime, ks, x, h):
    fd = [[h * y_prime(x - (4 - i) * h, ks[i]) for i in range(5)]]
    for _ in range(4):
        fd.append([y2 - y1 for y1, y2 in zip(fd[-1], fd[-1][1:])])
    return fd


def adams(y_prime, xs, ys, N, h):
    xs, ys = list(xs), list(ys)

    for i in range(N):
        fd = adams_fd(y_prime, ys[-5:], xs[-1], h)
        xs.append(xs[-1] + h)
        ys.append(ys[-1] + fd[0][4] + fd[1][3] / 2 + 5 * fd[2][2] / 12 + 3 * fd[3][1] / 8 + 251 * fd[4][0] / 720)

    return xs, ys


def compare(header, xs, ys, sol):
    print(
        header,
        tabulate(zip(range(len(xs)), xs, ys, [abs(y - sol(x)) for x, y in zip(xs, ys)]),
                 headers=['#', 'x', 'y', 'R'], floatfmt='.8f'),
        sep='\n', end='\n\n')

def main():
    print(
        "Лабораторная работа №7",
        "Численное решение задачи Коши",
        "-----------------------------",
        sep='\n', end='\n\n')

    print(
        y_prime.__doc__,
        "Задача Коши: y({0}) = {1}".format(x_0, y_0),
        "Решение задачи Коши: " + sol.__doc__,
        sep='\n', end='\n\n')

    #N = input_value("Введите число шагов N: ",
    #                value_type=int, check=lambda N: N >= 0)
    #h = input_value("Введите h (h > 0): ", check=lambda h: h > 0)
    print("Введите число шагов N и шаг h > 0:")
    N, h = list(map(float, input().split()))
    N = int(N)
    print()

    xs, ys = taylor(taylor_coeffs, x_0 - 2 * h, N + 3, h)
    compare("Метод разложения в ряд Тейлора", xs, ys, sol)

    xs, ys = euler1(y_prime, x_0, y_0, N, h)
    compare("Метод Эйлера #1", xs, ys, sol)
    xs, ys = euler2(y_prime, x_0, y_0, N, h)
    compare("Метод Эйлера #2", xs, ys, sol)
    xs, ys = euler3(y_prime, x_0, y_0, N, h)
    compare("Метод Эйлера #3", xs, ys, sol)

    xs, ys = runge_kutta(y_prime, x_0, y_0, N, h)
    compare("Метод Рунге-Кутты", xs, ys, sol)

    xs = [x_0 - (2 - i) * h for i in range(5)]
    ys = [taylor_eval(taylor_coeffs, x, 0) for x in xs]

    xs, ys = adams(y_prime, xs, ys, N - 2, h)
    compare("Метод Адамса", xs, ys, sol)


if __name__ == '__main__':
    main()
