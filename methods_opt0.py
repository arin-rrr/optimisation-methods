'''
Методы оптимизации 0-ого порядка одномерной функции
'''

import math


# метод пассивного поиска
def passive_search(a, b, eps, func):
    # задаём количество разбиений - здесь учитывается точность
    n = math.ceil((b - a) / eps)

    # счётчик для обращения к правой части
    k = 0

    # задаём точки, разделяющие отрезок [-5, -4]
    x = []
    for i in range(0, n + 1):
        x.append(a + i * (b - a) / n)

    y = []
    for x_i in x:
        y.append(float(func(x_i)))
        k += 1

    index_y_min = y.index(min(y))
    return x[index_y_min], y[index_y_min], k


# метод дихотомии
def dichotomy(a, b, eps, delta, func):
    a_curr = a
    b_curr = b
    c_curr = (a_curr + b_curr) / 2 - delta / 2
    d_curr = (a_curr + b_curr) / 2 + delta / 2

    # счётчик обращений к правой части
    k = 0

    while abs(b_curr - a_curr) >= 2 * eps:
        if func(c_curr) < func(d_curr):
            b_curr = d_curr
        elif func(c_curr) > func(d_curr):
            a_curr = c_curr
        else:
            a_curr = min(c_curr, d_curr)
            b_curr = max(c_curr, d_curr)
        c_curr = (a_curr + b_curr) / 2 - delta / 2
        d_curr = (a_curr + b_curr) / 2 + delta / 2
        k += 2

    # минимум - середина [a_curr, b_curr]
    return (a_curr + b_curr) / 2, float(func((a_curr + b_curr) / 2)), k


# метод золотого сечения
def golden_ratio(a, b, eps, func):
    a_curr = a
    b_curr = b
    c_curr = (3 / 2 - (5 ** 0.5) / 2) * (b_curr - a_curr) + a_curr
    d_curr = ((5 ** 0.5) / 2 - 1 / 2) * (b_curr - a_curr) + a_curr

    fc = func(c_curr)
    fd = func(d_curr)

    # счётчик обращений к правой части
    k = 0

    while abs(b_curr - a_curr) >= 2 * eps:
        if fc <= fd:
            b_curr = d_curr
            d_curr = c_curr
            fd = fc

            c_curr = (3 / 2 - (5 ** 0.5) / 2) * (b_curr - a_curr) + a_curr
            fc = func(c_curr)

            k += 1
        else:
            a_curr = c_curr
            c_curr = d_curr

            fc = fd

            d_curr = ((5 ** 0.5) / 2 - 1 / 2) * (b_curr - a_curr) + a_curr
            fd = func(d_curr)

            k += 1

    return (a_curr + b_curr) / 2, func((a_curr + b_curr) / 2), k

# метод Фибоначчи
def fibonacci(n):
    if n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci_method(a, b, eps, func):
    # выбираем n - количество итераций
    n = 1
    while True:
        if fibonacci(n) >= 11 * (b - a) / (20 * eps):
            break
        else:
            n += 1
    n -= 1

    a_curr = a
    b_curr = b

    # счётчик обращений к правой части функции
    k = 0

    for k in range(1, n):
        c_curr = a_curr + (b_curr - a_curr) * (fibonacci(n - k) / fibonacci(n + 2 - k))
        d_curr = a_curr + (b_curr - a_curr) * (fibonacci(n + 1 - k) / fibonacci(n + 2 - k))

        if func(c_curr) < func(d_curr):
            b_curr = d_curr
            k += 2
        else:
            a_curr = c_curr
            k += 2
    return (a_curr + b_curr) / 2, func((a_curr + b_curr) / 2), k
