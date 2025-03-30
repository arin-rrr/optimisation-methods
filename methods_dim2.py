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