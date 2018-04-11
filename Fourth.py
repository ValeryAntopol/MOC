import texttable as tt
from Third import *
from numpy.Polynominal import legendre



def main():
    print("f(x) = sin(x) + x^2")
    print("Введите a, b, m:")
    a, b, m = map(float, input().split())
    m = int(m)

    h, xs = nodes(a, b, m)
    ys = lmap(f, xs)

    print_table(a, b, h, xs, ys)
    eps = 1
    while True:
        print("Введите допустимую погрешность:")
        eps = float(input())
        if 0 < eps:
            break
+
        print("Допустимая погрешность должна быть положительным числом")
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
                for i in range(n//2, len(xs)-n//2):
                    if abs(xs[i]-x) < dmin:
                        middle = i
                        dmin = abs(xs[i]-x)


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
