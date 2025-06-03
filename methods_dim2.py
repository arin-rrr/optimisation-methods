import numpy as np
from methods_opt0 import golden_ratio

# метод покоординатного спуска
def coord_descent(x0, eps, func):
    x = np.array(x0, dtype=float)
    dim = len(x0)
    k = 0
    while True:
        curr_x = x.copy()
        for i in range(dim):
            def func_i(x_i):  # определеили функцию, которая зафиксирована по коор. кроме i
                x_temp = x.copy()
                x_temp[i] = x_i
                return func(x_temp[0], x_temp[1])

            x[i] = golden_ratio(-2, 2, eps, func_i)[0]

        if np.linalg.norm(x - curr_x) < eps and np.linalg.norm(func(x[0], x[1]) - func(curr_x[0], curr_x[1])) < eps:
            k += 1
            if k >= 5:
                break
    return x, func(x[0], x[1])


# покоординатный спуск с модификацией шага
def grad_descent_step_split(x0, func, df, alpha=1, eps=10 ^ (-6), lam=0.1, sigma=0.1):
    x = np.array(x0)

    for _ in range(1000):
        # считаю градиент в существующей точке
        gradient = np.array(df(x[0], x[1]))

        # вдруг в начальной точке выполнен критерий остановки
        if np.linalg.norm(gradient) < eps:
            break

        # выполняем дроблемние шага
        while True:
            x_new = x - alpha * gradient
            if func(x_new[0], x_new[1]) - func(x[0], x[1]) <= -sigma * alpha * np.linalg.norm(df(x[0], x[1])) ** 2:
                break
            alpha *= lam

        x = x_new
    return x

# МНГС
def mngs(x0, func, df, eps=10e-6, fast_grad=False):
    x = np.array(x0, dtype=np.float64)
    if not fast_grad:
        for _ in range(1000):
            gradient = np.array(df(x[0], x[1]), dtype=np.float64)

            # проверка критерий остановки
            if np.linalg.norm(gradient) <= eps:
                break

            # функция для нахождения оптимального альфа
            def f_min_alpha(alpha):
                x_new = x - alpha * gradient
                return func(x_new[0], x_new[1])

            alpha_opt = golden_ratio(0, 1, 10e-8, f_min_alpha)

            x -= alpha_opt[0] * gradient
        return x

    else:
        for _ in range(2):
            gradient = np.array(df(x[0], x[1]), dtype=np.float64)

            # проверка критерий остановки
            if np.linalg.norm(gradient) <= eps:
                break

            # функция для нахождения оптимального альфа
            def f_min_alpha(alpha):
                x_new = x - alpha * gradient
                return func(x_new[0], x_new[1])

            alpha_opt = golden_ratio(0, 1, 0.1, f_min_alpha)

            x -= alpha_opt[0] * gradient

        return x

# не помню
def fast_grad_p(x_begin, func, df, eps=10 ** (-6)):
    x = np.array(x_begin)

    for _ in range(1000):
        y0 = mngs(x, func, df, fast_grad=True)

        if np.linalg.norm(df(x[0], x[1])) <= eps:
            break

        # определяем направление
        direction = y0 - x

        def f_min_alpha(alpha):
            return func(x[0] + alpha * direction[0], x[1] + alpha * direction[1])

        alpha_opt = golden_ratio(0, 1, 0.1, f_min_alpha)[0]

        x = x + alpha_opt * direction
    return x


# овражный метод
def ovr_method(x_begin, func, df, delta=0.1, eps=10 ** (-4)):
    x = np.array(x_begin)

    for _ in range(1000):
        x_tilde = x + delta * np.random.randn(2)

        def grad_step(x):
            gradient = np.array(df(x[0], x[1]))

            def f_min_alpha(alpha):
                return func(x[0] - alpha * gradient[0], x[1] - alpha * gradient[1])

            alpha_opt = golden_ratio(0, 1, 0.1, f_min_alpha)[0]
            return x - alpha_opt * gradient

        y = grad_step(x)
        y_tilde = grad_step(x_tilde)

        direction = y_tilde - y

        # минимизируем вдоль направления
        def f_alpha(alpha):
            return func(y[0] + alpha * direction[0], y[1] + alpha * direction[1])

        alpha_opt = golden_ratio(0, 1, 0.01, f_alpha)[0]
        x_new = y + alpha_opt * direction

        if np.linalg.norm(df(x_new[0], x_new[1])) <= eps:
            return x_new
        x = x_new


# метод ньютона
def method_newton(x_begin, df, ddf, eps=0.01):
    x_curr = x_begin
    while np.linalg.norm(df(x_curr[0], x_curr[1])) > eps:
        inv_matrix = np.linalg.inv(ddf(x_curr[0], x_curr[1]))
        first_pr = df(x_curr[0], x_curr[1])
        x_curr -= inv_matrix @ first_pr
    return x_curr

# модифицированный метод ньютона
def modif_method_newton(x_begin, func, df, ddf, eps=0.001):
    x_curr = x_begin
    while np.linalg.norm(df(x_curr[0], x_curr[1])) > eps:
        inv_matrix = np.linalg.inv(ddf(x_curr[0], x_curr[1]))
        first_pr = df(x_curr[0], x_curr[1])

        def f_alpha(alpha):
            new_x = x_curr - alpha * (inv_matrix @ first_pr)
            return func(new_x[0], new_x[1])

        alpha_opt = golden_ratio(0, 1, 0.1, f_alpha)[0]
        x_curr -= alpha_opt * (inv_matrix @ first_pr)
    return x_curr

# квази ньютоновский метод
def quasi_newton_method(func, df, x0, eps=0.001):
    x_curr = x0
    H = np.eye(2)

    for k in range(1000):
        if np.linalg.norm(df(x_curr[0], x_curr[1])) < eps:
            break

        def f_alpha(alpha):
            return func((x_curr - alpha * (H @ df(x_curr[0], x_curr[1])))[0],
                        (x_curr - alpha * (H @ df(x_curr[0], x_curr[1])))[1])

        alpha_opt = golden_ratio(0, 1, 0.1, f_alpha)[0]

        x_prev = x_curr
        grad_prev = df(x_curr[0], x_curr[1])

        x_curr -= alpha_opt * (H @ df(x_curr[0], x_curr[1]))

        delta = x_curr - x_prev
        gamma = df(x_curr[0], x_curr[1]) - grad_prev

        # обновление матрицы H
        if (k + 1) % 2 != 0:
            numerator = np.outer(delta - H @ gamma, delta - H @ gamma)
            denominator = (delta - H @ gamma).T @ gamma

            if abs(denominator) > 1e-10:
                H = H + numerator / denominator
        else:
            H = np.eye(2)

    return x_curr

# метод Флетчера
def fletcher_method(x_begin, func, df, eps=0.001):
    x_curr = x_begin
    d_curr = -df(x_begin[0], x_begin[1])
    x = [x_curr]
    d = [d_curr]
    for k in range(1000):
        if np.linalg.norm(d_curr) < eps:
            break

        def f_alpha(alpha):
            return func((x_curr+alpha*d_curr)[0], (x_curr+alpha*d_curr)[1])

        alpha_opt = golden_ratio(0, 1, 0.1, f_alpha)[0]
        if (k+1) % 2 == 0:
            x_curr += alpha_opt*d_curr
            d_curr = -df(x_curr[0], x_curr[1])
            x.append(x_curr)
            d.append(d_curr)
        else:
            x_curr += alpha_opt*d_curr
            beta = (np.linalg.norm(df(x_curr[0], x_curr[1]))/np.linalg.norm(df(x[-1][0], x[-1][1])))**2
            x.append(x_curr)

            d_curr = -df(x_curr[0], x_curr[1]) + beta*d[-1]
            d.append(d_curr)
    return x_curr

