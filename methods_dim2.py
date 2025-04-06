import numpy as np
from methods_opt0 import golden_ratio


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


def mngs(x0, func, df, eps=0.001, fast_grad=False):
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

            alpha_opt = golden_ratio(0, 1, 0.1, f_min_alpha)

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


def fast_grad_p(x_begin, func, df, eps=10**(-6)):
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


def ovr_method(x_begin, func, df, delta=0.1, eps=10**(-4)):
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
        x_new = y + alpha_opt*direction

        if np.linalg.norm(df(x_new[0], x_new[1])) <= eps:
            return x_new
        x = x_new
    # return x