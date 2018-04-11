from Third import *
import math


def f(x):
    return x * x * x   + math.sin(x)


def w(x):
    return 1


def F(x):
    return 1 / 4.0 * x * x * x * x   + math.cos(x)


def middleRectangles(a, h, m, func):
    sum = 0
    for k in range(m):
        sum += func(a + h / 2 + k * h)

    return sum * h


def trapezi(a, h, m, func):
    sum = 0
    for i in range(1, m):
        sum += f(a + h * i)
    return h / 2 * (func(a) + 2 * sum + func(a + h * m))


def simpson(a, h, m, func):
    sum_even = 0
    sum_odd = 0
    h /= 2
    for i in range(m):
        sum_odd += func(a + h * (2 * i + 1))
        sum_even += func(a + h * 2 * (i + 1))

    return h * (func(a) + 4 * sum_odd + 2 * sum_even - func(a + 2 * m * h)) / 3


def main():
    fstr = "f(x) = x^2"  # + sin(x)"
    print(fstr)
    print("Введите пределы интегрирования A, B и число m:")
    a, b, m = lmap(float, input().split())
    m = int(m)
    h = (b - a) / m
    J = F(b) - F(a)

    print("Результаты вычисления:")
    print("%.10f - интеграл" % J)
    print("%.10f - составная формула средних треугольников" % middleRectangles(a, h, m, f))
    print("%.10f - составная формула трапеций" % trapezi(a, h, m, f))
    print("%.10f - составная формула Симпсона" % simpson(a, h, m, f))

    print("Абсолютные погрешности:")
    print("%.10f - составная формула средних треугольников" % math.fabs(J - middleRectangles(a, h, m, f)))
    print("%.10f - составная формула трапеций" % math.fabs(J - trapezi(a, h, m, f)))
    print("%.10f - составная формула Симпсона" % math.fabs(J - simpson(a, h, m, f)))


if __name__ == "__main__":
    main()
