import numpy as np
from copy import deepcopy
import texttable as tt


def n_rows(mat):
    return len(mat)


def n_cols(mat):
    return len(mat[0])


def gauss_sub_row(mat, src_row, alpha, dst_row):
    for col in range(n_cols(mat)):
        mat[dst_row][col] += alpha * mat[src_row][col]


def gauss_mult_row(mat, row, alpha):
    for col in range(n_cols(mat)):
        mat[row][col] *= alpha


def add_col(mat, col):
    for row in range(n_rows(mat)):
        mat[row].append(col[row])


def print_mat(mat):
    for row in mat:
        print(row)


def dot(v1, v2):
    return np.sum([x * y for x, y in zip(v1, v2)])


def vlen2(v):
    return dot(v, v)


def vlen(v):
    return np.sqrt(vlen2(v))


def identity(mat):
    mat2 = deepcopy(mat)
    for row in range(len(mat)):
        for col in range(len(mat)):
            if row == col:
                mat2[row][col] = np.float32(1)
            else:
                mat2[row][col] = np.float32(0)
    return mat2


def gauss(a, b, eps=0):
    mat = deepcopy(a)
    add_col(mat, b)
    i = 1
    for row in range(n_rows(mat)):
        if np.abs(mat[row][row]) < eps:
            print("Предупреждение: деление на %f" % (mat[row][row]))

        #mat[row] = np.divide(mat[row], mat[row][row])
        divisor = mat[row][row]
        print("Ведущий элемент %f" % divisor)
        for col in range(row+1, n_cols(mat)):
            mat[row][col] = mat[row][col] / divisor

        for row2 in range(row + 1, n_rows(mat)):
            c = mat[row2][row]
            for col in range(row + 1, n_cols(mat)):
                mat[row2][col] -= mat[row][col] * c

        print("Расширенная метрица системы, шаг %i" % i)
        print_mat(mat)
        i += 1

    xs = [np.float32(0) for _ in range(n_rows(mat))]
    for x in reversed(range(len(xs))):
        xs[x] = mat[x][-1]
        for x2 in range(x + 1, len(xs)):
            xs[x] -= xs[x2] * mat[x][x2]

    return xs


def gauss_main_col(a, b, eps=0):
    mat = deepcopy(a)
    add_col(mat, b)
    for row in range(n_rows(mat)):
        max_ind = row
        for row2 in range(row + 1, n_rows(mat)):
            if np.abs(mat[row2][row]) > np.abs(mat[max_ind][row]):
                max_ind = row2

        tmp = mat[row]
        mat[row] = mat[max_ind]
        mat[max_ind] = tmp
        if np.abs(mat[row][row]) < eps:
            print("Предупреждение: деление на %f" % (mat[row][row]))

        mat[row] = np.divide(mat[row], mat[row][row])
        for row2 in range(row + 1, n_rows(mat)):
            c = mat[row2][row]
            for col in range(n_cols(mat)):
                mat[row2][col] -= mat[row][col] * c

    xs = [np.float64(0) for _ in range(n_rows(mat))]
    for x in reversed(range(len(xs))):
        xs[x] = mat[x][-1]
        for x2 in range(x + 1, len(xs)):
            xs[x] -= xs[x2] * mat[x][x2]

    return xs


def gauss_main_col_det(a):
    mat = deepcopy(a)
    det = np.float32(1)
    for row in range(n_rows(mat)):
        max_ind = row
        for row2 in range(row, n_rows(mat)):
            if np.abs(mat[row2][row]) > np.abs(mat[max_ind][row]):
                max_ind = row2

        tmp = mat[row]
        mat[row] = mat[max_ind]
        mat[max_ind] = tmp
        det *= mat[row][row]
        print("множителдь определителя %f" % mat[row][row])
        mat[row] = np.divide(mat[row], mat[row][row])
        if row != max_ind:
            det *= np.float32(-1)



        for row2 in range(row + 1, n_rows(mat)):
            c = mat[row2][row]
            for col in range(n_cols(mat)):
                mat[row2][col] -= mat[row][col] * c

    return det


def calc_bounds_for_lambda(mat):
    m, M = np.float32(-1), np.float32(-1)
    for row in range(n_rows(mat)):
        m_sum = np.float32(0)
        for col in range(n_cols(mat)):
            if col != row:
                m_sum += np.abs(mat[row][col])

        if m == -1 or m > mat[row][row] - m_sum:
            m = mat[row][row] - m_sum

        if M == -1 or M < mat[row][row] + m_sum:
            M = mat[row][row] + m_sum

    return m, M


def mat_norm(mat):
    return np.max([np.sum(np.abs(row)) for row in mat])


def vec_norm(v):
    return np.max(np.abs(v))


def fillq(l):
    q = [[] for _ in range(l)]
    q[0] = [1]
    for t in range(1, l):
        q[t] = [[] for _ in range(2 ** t)]
        for i in range(len(q[t])//2):
            q[t][2*i] = q[t-1][i]
            q[t][2*i+1] = 2 * len(q[t]) - q[t][2*i]

    return q[-1]


def powers_method(A, eps):
    x_old = [0 for _ in A]
    x_new = [1 for _ in A]
    lambda_old, lambda_new = 0, 1
    n = 0
    while abs(lambda_new - lambda_old) > eps:
        n += 1
        x_old = x_new
        lambda_old = lambda_new
        x_new = np.matmul(A, x_new)
        good_index = len(x_old)
        for i in range(len(x_old)):
            if x_old[i] > eps:
                lambda_new = x_new[i] / x_old[i]
                good_index = i
                break

        if good_index == len(x_old):
            print("собственное число примерно 0")
            return

        x_new = np.divide(x_new, vec_norm(x_new))

    return x_new, lambda_new, n


def dots_method(A, eps):
    x_old = [0.5 for _ in A]
    x_new = [1 for _ in A]
    lambda_old, lambda_new = 0, 1
    n = 0
    while abs(lambda_new - lambda_old) > eps:
        n += 1
        x_old = x_new
        lambda_old = lambda_new
        x_new = np.matmul(A, x_new)
        lambda_new = dot(x_new, x_old) / dot(x_old, x_old)
        x_new = np.divide(x_new, vec_norm(x_new))

    return x_new, lambda_new, n


def jacobi_method(A, eps):
    Ak = A
    X = identity(A)
    n = 0
    while True:
        n += 1
        ik, jk = 0, 1
        for row in range(n_rows(Ak)):
            for col in range(row + 1, n_cols(Ak)):
                if abs(Ak[row][col]) >= abs(Ak[ik][jk]):
                    ik = row
                    jk = col

        if abs(Ak[ik][jk]) < eps:
            break

        d = np.sqrt((Ak[ik][ik] - Ak[jk][jk]) * (Ak[ik][ik] - Ak[jk][jk]) + 4 * Ak[ik][jk] * Ak[ik][jk])
        c = np.sqrt((1 + abs(Ak[ik][ik] - Ak[jk][jk]) / d) / 2)
        s = np.sign(Ak[ik][jk] * (Ak[ik][ik] - Ak[jk][jk])) * np.sqrt((1 - abs(Ak[ik][ik] - Ak[jk][jk]) / d) / 2)

        Ik = deepcopy(get_col(X, ik))
        Jk = deepcopy(get_col(X, jk))

        for i in range(len(X)):
            X[i][ik] = c * Ik[i] + s * Jk[i]
            X[i][jk] = -s * Ik[i] + c * Jk[i]

        Ak_new = deepcopy(Ak)
        for i in range(len(Ak)):
            Ak_new[i][ik] = Ak_new[ik][i] = c * Ak[i][ik] + s * Ak[i][jk]
            if i != ik and i != jk:
                Ak_new[i][jk] = Ak_new[jk][i] = -s * Ak[i][ik] + c * Ak[i][jk]

        Ak_new[ik][ik] = c ** 2 * Ak[ik][ik] + 2 * c * s * Ak[ik][jk] + s ** 2 * Ak[jk][jk]
        Ak_new[jk][jk] = s ** 2 * Ak[ik][ik] - 2 * c * s * Ak[ik][jk] + c ** 2 * Ak[jk][jk]
        Ak_new[ik][jk] = Ak_new[jk][ik] = 0
        Ak = Ak_new


    return X, Ak, n


def get_col(mat, col):
    return [row[col] for row in mat]


def solve_3diag(n, a, b, c, d):
    m = [0.0 for _ in a]
    k = [0.0 for _ in a]
    y = [0.0 for _ in a]
    m[1] = -c[0] / b[0]
    k[1] = d[0] / b[0]
    for i in range(1, n):
        m[i + 1] = -c[i] / (a[i] * m[i] + b[i])
        k[i + 1] = (d[i] - a[i] * k[i]) / (a[i] * m[i] + b[i])

    print("||m|| = {0}".format(vec_norm(m)))

    y[n] = (d[n] - a[n] * k[n]) / (a[n] * m[n] + b[n])
    for i in range(n, 0, -1):
        y[i-1] = m[i] * y[i] + k[i]

    r = [0 for _ in a]
    r[0] = b[0] * y[0] + c[0] * y[1] - d[0]
    for i in range(1, n):
        r[i] = a[i] * y[i - 1] + b[i] * y[i] + c[i] * y[i + 1] - d[i]

    r[n] = a[n] * y[n - 1] + b[n] * y[n] - d[n]

    print("||Ay-d|| = {0}".format(vec_norm(r)))
    return y


def solve_3diag_quiet(n, a, b, c, d):
    m = [0.0 for _ in a]
    k = [0.0 for _ in a]
    y = [0.0 for _ in a]
    m[1] = -c[0] / b[0]
    k[1] = d[0] / b[0]
    for i in range(1, n):
        m[i + 1] = -c[i] / (a[i] * m[i] + b[i])
        k[i + 1] = (d[i] - a[i] * k[i]) / (a[i] * m[i] + b[i])

    y[n] = (d[n] - a[n] * k[n]) / (a[n] * m[n] + b[n])
    for i in range(n, 0, -1):
        y[i-1] = m[i] * y[i] + k[i]

    return y


def create_kernel(n):
    alpha = []
    betha = []
    alpha.append(lambda x: 1)
    betha.append(lambda x: 1 / 5)
    p = 1

    for i in range(1, n):
        p *= i
        alpha.append(lambda x, i=i, p=p: x ** i / p)
        betha.append(lambda x, i=i: x ** i / 5)

    return alpha, betha


def gauss_int(a, b, m, f1, f2):
    res = 0.0
    h = (b - a) / m
    for i in range(m):
        res += f1(a + h / 2 + h * i) * f2(a + h / 2 + h * i)

    return res * h


def create_lin_system(left, right, m, alphas, betas, f, n):
    A = [[0.0 for _ in range(n)] for _ in range(n)]
    b = [0.0 for _ in range(n)]
    for i in range(n):
        b[i] = gauss_int(left, right, m, betas[i], f)
        for j in range(n):
            A[i][j] = gauss_int(left, right, m, alphas[j], betas[i])
            if i == j:
                A[i][j] += 1.0

    return A, b


def solve_kernels_n(left, right, m, f_func, q, N):
    alphas, betas = create_kernel(q)
    A, B = create_lin_system(left, right, m, alphas, betas, f_func, q)
    C = gauss_main_col(A, B)
    u = lambda x: f_func(x) - dot(C, [f(x) for f in alphas])
    h = (right - left) / N
    return [u(left + i * h) for i in range(N+1)]


def get_G(nodes, f_func):
    g = []
    for no in nodes:
        g.append(f_func(no))

    return g


def get_D(n, nodes, coeffs, g, K_func):
    D = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            d = 0
            if j == k:
                d = 1
            D[j][k] = d + coeffs[k] * K_func(nodes[j], nodes[k])

    return D.tolist()


def solve_mmk(n, nodes, coeffs, K_func, f_func, a, b, N):
    g = [f_func(node) for node in nodes]
    D = get_D(n, nodes, coeffs, g, K_func)
    u = gauss_main_col(D, g)

    def un(t):
        res = f_func(t)
        for i in range(n):
            res -= coeffs[i] * K_func(t, nodes[i]) * u[i]

        return res

    return [un(a + (b - a) * i / N) for i in range(N + 1)]


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


def explicit_scheme(n, m, g, x, t, alpha, beta, tau, f, h, U):
    print("Явная схема")
    u = [[0.0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(n + 1):
        u[0][i] = g(x[i])

    for k in range(1, m + 1):
        u[k][0] = alpha(t[k])
        u[k][n] = beta(t[k])
        for i in range(1, n):
            u[k][i] = tau * (u[k - 1][i + 1] + u[k - 1][i - 1]) / h ** 2 + u[k - 1][i] * (1 - 2 * tau / h ** 2) + tau * f(x[i], t[k - 1])

    while True:
        print("Введите слой от 0 до {0} или -1 чтобы продолжить: ".format(m))
        k = int(input())
        if k >= 0:
            if k <= m:
                print("t = {0}".format(t[k]))
                u0 = [U(x[i], t[k]) for i in range(n + 1)]
                print("{0:<15}{1:<15}".format("Точное", "Явная схема"))
                for i in range(n + 1):
                    print("{0:< 15.6}{1:< 15.6}".format(u0[i], u[k][i]))
            else:
                print("Слой должен быть <= {0}".format(m))
        else:
            if k == -1:
                return
            else:
                print("Неверный ввод")


def implicit_scheme(n, m, g, x, t, alpha, beta, tau, f, h, U):
    print("Неявная схема")
    u = [[0.0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(n + 1):
        u[0][i] = g(x[i])

    a = [0.0 for _ in range(n + 1)]
    b = [0.0 for _ in range(n + 1)]
    c = [0.0 for _ in range(n + 1)]
    d = [0.0 for _ in range(n + 1)]
    a[0] = 0
    b[0] = 1
    c[0] = 0
    a[n] = 0
    b[n] = 1
    c[n] = 0
    for i in range(1, n):
        a[i] = -tau / h ** 2
        b[i] = 1 + 2 * tau / h ** 2
        c[i] = -tau / h ** 2

    for k in range(1, m + 1):
        d[0] = alpha(t[k])
        d[n] = beta(t[k])
        for i in range(1, n):
            d[i] = tau * f(x[i], t[k]) + u[k - 1][i]

        u[k] = solve_3diag_quiet(n, a, b, c, d)


    while True:
        print("Введите слой от 0 до {0} или -1 чтобы продолжить: ".format(m))
        k = int(input())
        if k >= 0:
            if k <= m:
                print("t = {0}".format(t[k]))
                u0 = [U(x[i], t[k]) for i in range(n + 1)]
                print("{0:<15}{1:<15}".format("Точное", "Неявная схема"))
                for i in range(n + 1):
                    print("{0:<15.6}{1:<15.6}".format(u0[i], u[k][i]))
            else:
                print("Слой должен быть <= {0}".format(m))
        else:
            if k == -1:
                return
            else:
                print("Неверный ввод")


def main_first():
    a = [[np.float32(6.8704E-06), np.float32(-73.592E-03), np.float32(4.34112)],
         [np.float32(6.2704E-03), np.float32(-0.73592),   np.float32(1.57112)],
         [np.float32(0.90704),     np.float32(0.3208),   np.float32(1.02112)]]
    b = [np.float32(3.09008),
         np.float32(1.70505),
         np.float32(2.04310)]

    ab = deepcopy(a)
    add_col(ab, b)
    print("Точные методы решения систем линейных алгебраических уравнений. Метод Гаусса.")
    print("Решить СЛАУ Ax=b")
    print("A|b=")
    print_mat(ab)
    print("Введите ограничение, при котором будет предупреждение при делении, eps = ")
    eps = np.float32(input())
    print()

    print("Метод Гаусса (схема единственного деления)")
    xs1 = gauss(a, b, eps)
    err1 = np.subtract(np.matmul(a, xs1), b)
    xs2 = gauss_main_col(a, b, eps)
    err2 = np.subtract(np.matmul(a, xs2), b)

    print("x=")
    print_mat(xs1)
    print("Модуль невязки:")
    print(vec_norm(err1))
    print()
    print("Метод Гаусса с выбором главного элемента по столбцу")
    print("x=")
    print_mat(xs2)
    print("Модуль невязки:")
    print(vec_norm(err2))
    print()
    print("Определитель матрицы A:")
    print(gauss_main_col_det(a))


def main_second():
    a = [[np.float64(2.20219), np.float64(0.33266), np.float64(0.16768)],
         [np.float64(0.33266), np.float64(3.17137), np.float64(0.54055)],
         [np.float64(0.16768), np.float64(0.54055), np.float64(4.92343)]]
    b = [np.float64(2.17523),
         np.float64(6.58335),
         np.float64(6.36904)]

    ab = deepcopy(a)
    add_col(ab, b)

    print("Решение линейных алгебраических уравнений итерационными методами")
    print("Решить СЛАУ Ax=b")
    print("A|b=")
    print_mat(ab)

    m, M = calc_bounds_for_lambda(a)
    print("По теореме о кргуах Гершгорина, все собственные числа lambda")
    print("m = %f <= lambda <= %f = M" % (m, M))
    print()

    print("Методом Гаусса с выбором главного элемента по столбцу:")
    x_gauss = gauss_main_col(a, b)
    err = np.subtract(np.matmul(a, x_gauss), b)

    print("Ax=")
    print_mat(np.matmul(a, x_gauss))
    print("b=")
    print_mat(b)
    print("||Ax - b|| = {0:<.10}".format(vec_norm(err)))
    print()

    print("Введите точность, eps = ")
    eps = np.float64(input())
    print()

    alpha = np.float32(2) / (m + M)
    print("Метод простой итерации с оптимальнымм параметром:")
    print("Оптимальный парметр alpha = 2 / (M + m) = %f" % alpha)
    print("Новая система: x = B_alpha * x + c_alpha")
    b_alpha = np.subtract(identity(a), np.multiply(alpha, a)).tolist()
    c_alpha = np.multiply(alpha, b).tolist()
    bc = deepcopy(b_alpha)
    add_col(bc, c_alpha)
    print_mat(bc)
    norm_b = mat_norm(b_alpha)
    print("||B_alpha|| = %f" % norm_b)
    x_old = [np.float32(0) for _ in b]
    x = np.add(np.matmul(b_alpha, x_old), c_alpha).tolist()
    k_iter = 0
    delta = vec_norm(x)
    mult = delta
    k_apr = int(np.log(eps * (1 - norm_b) / delta) / np.log(norm_b))
    print("{0:5}{1:15}{2:15}{3:15}".format("Шаг", "Фактическая", "Апостериорная", "Априорная"))

    def print_data(iter, fact, aposterior, aprior):
        print("{0:<-5}{1:<-15.6}{2:<-15.6}{3:<-15.6}".format(iter, fact, aposterior, aprior))

    mult *= norm_b
    print_data(k_iter,
               vec_norm(np.subtract(x_gauss, x)),
               mult / (np.float32(1) - norm_b),
               delta * norm_b / (np.float32(1) - norm_b))
    k_iter += 1
    while delta > eps:
        x_old = x
        x = np.add(np.matmul(b_alpha, x_old), c_alpha).tolist()
        delta = vec_norm(np.subtract(x_old, x))
        mult *= norm_b
        print_data(k_iter,
                   vec_norm(np.subtract(x_gauss, x)),
                   delta * norm_b / (np.float32(1) - norm_b),
                   mult / (np.float32(1) - norm_b))
        k_iter += 1

    print("Априорная оценка k = {0}, фактическое число итераций = {0}".format(k_apr, k_iter))
    print("x=")
    print_mat(x)
    print("||Ax - b|| = {0}".format(vec_norm(np.subtract(np.matmul(a, x), b))))
    print("||B_alpha * x + c_alpha - x|| =  {0}".format(vec_norm(np.subtract(np.add(np.matmul(b_alpha, x), c_alpha), x))))
    print()
    print("Метод  простой итерации с Чебышевским набором параметров:")
    d = np.sqrt(M / m) * np.log(2 / eps) / 2
    l = int(np.ceil(np.log(d)/np.log(2)))
    K = 2 ** l
    print("d = {0} l = {1} K = {2}".format(d, l, K))
    q = fillq(l+1)
    tau = 0
    x = [0 for _ in b]
    cur_r = np.subtract(b, np.matmul(a, x))
    print("{0:5}{1:15}{2:15}".format("Шаг", "Фактическая", "||Ax(k)-b||"))
    for k in range(K):
        tau = 2 / ((M + m) - (M - m) * np.cos(q[k] * np.pi / 2 / K))
        x = np.add(x, np.multiply(tau, cur_r))
        cur_r = np.subtract(b, np.matmul(a, x))
        print("{0:<-5}{1:<-15.6}{2:<-15.6}".format(k, vec_norm(np.subtract(x_gauss, x)), vec_norm(cur_r)))

    print("x=")
    print_mat(x)


def main_third():
    print("Решение полной и частичной проблемы собственных значений.")
    A = [[-1.536984, -0.199070, 0.958551],
         [-0.199070, 1.177416, 0.069925],
         [0.958551, 0.069925, -1.51506]]
    eps = 1e-7
    print("A=")
    print_mat(A)

    print("Степенной метод - максимальное по модулю собственное число")
    x1, lambda1, n = powers_method(A, eps)
    print("Точность eps= {0}".format(eps))
    print("Количество итераций: {0}".format(n))
    print("lambda1= {0}".format(lambda1))
    print("x1=")
    print_mat(x1)
    print("Невязка:")
    print_mat(np.subtract(np.matmul(A, x1), np.multiply(x1, lambda1)))
    print()

    B = np.subtract(A, np.multiply(lambda1, identity(A)))
    print("Степенной метод - противоположная граница спектра")
    x2, lambda2, n = powers_method(B, eps)
    lambda2 += lambda1
    print("Точность eps= {0}".format(eps))
    print("Количество итераций: {0}".format(n))
    print("lambda2= {0}".format(lambda2))
    print("x2=")
    print_mat(x2)
    print("Невязка:")
    print_mat(np.subtract(np.matmul(A, x2), np.multiply(x2, lambda2)))
    print()

    lambda3 = A[0][0] + A[1][1] + A[2][2] - lambda1 - lambda2
    print("Степенной метод - третье собственное число")
    print("lambda3= {0}".format(lambda3))
    print()

    print("Метод скалярных произведений - максимальное по модулю собственное число")
    x1_2, lambda1_2, n = dots_method(A, eps)
    print("Точность eps= {0}".format(eps))
    print("Количество итераций: {0}".format(n))
    print("lambda1= {0}".format(lambda1_2))
    print("x1=")
    print_mat(x1_2)
    print("Невязка:")
    print_mat(np.subtract(np.matmul(A, x1_2), np.multiply(x1_2, lambda1_2)))
    print()

    print("Метод Якоби")
    X, Ak, n = jacobi_method(A, eps)
    print("Точность eps= {0}".format(eps))
    print("Количество итераций: {0}".format(n))
    print("Матрица с собственными числами:")
    print_mat(Ak)
    print("Матрица с собственными векторами в столбцах:")
    print_mat(X)
    Err = [[], [], []]
    for i in range(len(A)):
        col = get_col(X, i)
        add_col(Err, np.subtract(np.matmul(A, col), np.multiply(col, Ak[i][i])))

    print("Матрица с невязками:")
    print_mat(Err)


def main_fourth():
    def q_func(x):
        return np.log(x + 2)

    def r_func(x):
        return -x

    def f_func(x):
        return x + 2

    alpha = 0.6
    beta = 0.4
    left = 0.0
    right = 1.0

    alpha1 = alpha
    alpha2 = -1.0
    beta1 = beta
    beta2 = 1.0

    A = 0.0
    B = 0.0
    print("Разностный метод для обыкновенного дифференциального уравнения второго порядка. Метод прогонки.")
    print("Решить краевую задачу")
    print("Введите размер сетки, n=")
    n = int(input())
    p = [0.0 for _ in range(n + 1)]
    q = [0.0 for _ in range(n + 1)]
    r = [0.0 for _ in range(n + 1)]
    f = [0.0 for _ in range(n + 1)]
    x = [0.0 for _ in range(n + 1)]
    for i in range(n + 1):
        x[i] = left + i * (right - left) / n
        p[i] = 1.0
        q[i] = q_func(x[i])
        r[i] = r_func(x[i])
        f[i] = f_func(x[i])

    h = (right - left) / n
    a = np.subtract(p, np.multiply(q, h / 2.0))
    b = np.add(np.multiply(-2.0, p), np.multiply(h * h, r))
    c = np.add(p, np.multiply(q, h / 2.0))
    d = np.multiply(h * h, f)
    print("Схема первого порядка")
    a[0] = 0.0
    b[0] = h * alpha1 - alpha2
    c[0] = alpha2
    d[0] = h * A
    a[n] = -beta2
    b[n] = h * beta1 + beta2
    c[n] = 0.0
    d[n] = h * B
    y1 = solve_3diag(n, a, b, c, d)
    print()

    print("Схема второго порядка")
    a[0] = 0
    b[0] = 2 * h * alpha1 + alpha2 * (a[1] / c[1] - 3)
    c[0] = alpha2 * (b[1] / c[1] + 4)
    d[0] = 2 * h * A + alpha2 * d[1] / c[1]
    a[n] = -beta2 * (4 + b[n - 1] / a[n - 1])
    b[n] = 2 * h * beta1 + beta2 * (3 - c[n - 1] / a[n - 1])
    c[n] = 0
    d[n] = 2 * h * B - beta2 * d[n - 1] / a[n - 1]
    y2 = solve_3diag(n, a, b, c, d)
    print("||y2 - y1|| = {0}".format(vec_norm(np.subtract(y2, y1))))


def main_fifth():
    print("Численное решение интегрального уравнения Фредгольма 2-го рода. Метод механических квадратур.Метод замены ядра на вырожденное")

    def K_func(x, y):
        return np.exp(x * y) / 5.0

    def f_func(x):
        return 1 - x ** 2

    a = 0.0
    b = 1.0
    N = 10
    m = 1000

    print("Meтод замены ядра на вырожденное")
    alpha, beta = create_kernel(4)
    for i in range(1, 3):
        for j in range(1, 3):
            x = a + i * (b - a) / 2
            y = a + j * (b - a) / 2
            K0 = -K_func(x, y)
            for k in range(3):
                K0 += alpha[k](x) * beta[k](y)
            print("({0}, {1}) -- {2} -- {3}".format(x, y, abs(K0), abs(K0 + alpha[3](x) * beta[3](y))))



    xs = [a + (b - a) / N * i for i in range(N + 1)]
    ys3 = solve_kernels_n(a, b, m, f_func, 3, N)
    ys4 = solve_kernels_n(a, b, m, f_func, 4, N)

    print("max|u4(zk) - u3(zk)| = {0}".format(vec_norm(np.subtract(ys3, ys4))))
    print("{0:<15}{1:<15}{2:<15}".format("x", "y3", 'y4'))
    for i in range(N + 1):
        print("{0:<-15.6}{1:<-15.6}{2:<-15.6}".format(xs[i], ys3[i], ys4[i]))

    print("Метод механических квадратур")
    nodes3 = [-0.7745966692, 0, 0.7745966692]
    coeffs3 = [0.5555555556, 0.8888888889, 0.5555555556]
    for i in range(3):
        coeffs3[i] *= (b - a) / 2
        nodes3[i] = nodes3[i] * (b - a) / 2 + (b + a) / 2

    y_mmk3 = solve_mmk(3, nodes3, coeffs3, K_func, f_func, a, b, N)

    nodes4 = [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116]
    coeffs4 = [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]
    for i in range(4):
        nodes4[i] = nodes4[i] * (b - a) / 2 + (b + a) / 2
        coeffs4[i] *= (b - a) / 2

    y_mmk4 = solve_mmk(4, nodes4, coeffs4, K_func, f_func, a, b, N)
    print("{0:<15}{1:<15}{2:<15}".format("x", "y_mmk3", 'y_mmk4'))
    for i in range(N + 1):
        print("{0:<-15.6}{1:<-15.6}{2:<-15.6}".format(xs[i], y_mmk3[i], y_mmk4[i]))


def main_sixth():
    U = lambda x, t: np.exp(-4 * t) * np.cos(2 * x) + np.exp(-t) * (1 - x) * x
    f = lambda x, t: np.exp(-t) * (x ** 2 - x + 2)
    g = lambda x: np.cos(2*x) + (1 - x) * x
    alpha = lambda t: np.exp(-4 * t)
    beta = lambda t: np.exp(-4 * t) * np.cos(2)
    print("Введите N - число разбиений по х:")
    N = int(input())
    print("Введите Т - время, до которого рассматриваем:")
    T = float(input())
    print("Введите тау1 и тау2 длина промежутков разбиения для явного и неявного метода")
    print("tau1:")
    tau1 = float(input())
    print("tau2:")
    tau2 = float(input())
    h = 1.0 / N
    x = [i * h for i in range(N+1)]
    m1 = int(T / tau1)
    m2 = int(T / tau2)
    t1 = [k * tau1 for k in range(m1+1)]
    t2 = [k * tau2 for k in range(m2+1)]
    explicit_scheme(N, m1, g, x, t1, alpha, beta, tau1, f, h, U)
    implicit_scheme(N, m2, g, x, t2, alpha, beta, tau2, f, h, U)


if __name__ == "__main__":
    main_second()







