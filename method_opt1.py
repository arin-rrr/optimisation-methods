import math


def method_kasat(a, b, eps, func, df):

    func_k, df_k = 0, 0

    a_temp, b_temp = a, b
    c_temp = (func(b_temp) - func(a_temp) + a_temp * df(a_temp) - b_temp * df(b_temp))/(df(a_temp) - df(b_temp))
    func_k += 2
    df_k += 2

    while abs(df(c_temp)) >= eps:
        if df(c_temp) > 0:
            b_temp = c_temp
        else:
            a_temp = c_temp
        c_temp = (func(b_temp) - func(a_temp) + a_temp * df(a_temp) - b_temp * df(b_temp)) / (df(a_temp) - df(b_temp))
        func_k += 2
        df_k += 2

    return c_temp, func(c_temp), func_k, df_k


def method_hord(a, b, eps, func, df):
    # используем приближение для второй производной
    x0, x1 = a, b
    x2 = x1 - df(x1) * (x1 - x0)/ (df(x1) - df(x0))
    df_k = 2
    while abs(df(x2)) >= eps:
        x0 = x1
        x1 = x2
        x2 = x1 - df(x1) * (x1 - x0) / (df(x1) - df(x0))
        df_k += 2
    return x2, func(x2), df_k
