import math
from methods_opt0 import passive_search


def newton_rafson(a, eps, func, df, ddf):
    x_0 = a
    df_k = 0
    ddf_k = 0

    while abs(df(x_0)) > eps:
        df1 = df(x_0)
        df2 = ddf(x_0)
        x_0 -= df1/df2
        ddf_k += 1
        df_k += 1
    return x_0, func(x_0), df_k, ddf_k
