import numpy as np
from methods_dim2 import coord_descent
from scipy.optimize import minimize

'''
Реализация двух методов условной оптимизации - Куна-Такера и метод внешних штрафов
'''

def Kuhn_Tucker(func, ogran):
    '''
    :param func: функция
    :param df: проивзодные функции
    :param x0: начально еприближение
    :param ogran: список ограничений
    :param d_ogran: список производных ограничений
    :return: все решения (x1, x2, lambda) для каждой системы
    '''

    lambdas = []  # создаем кортежи лямбд, 0 - лямбда обнуляется, 1 - не обнуляется
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                for m in [0, 1]:
                    lambdas.append((i, j, k, m))

    all_solutions = []
    for l in lambdas:
        solve = [0, 0, 0, 0, 0, 0]  # x1, x2, lamb1, lamb2, lamb3, lamb4
        if l.count(1) == 0:
            A = np.array([
                [2, 0], [0, 18]
            ])
            b = np.array([12, 36])
            try:
                x = np.linalg.solve(A, b)
                solve[0], solve[1] = float(x[0]), float(x[1])
                all_solutions.append(solve)
            except np.linalg.LinAlgError:
                all_solutions.append(None)
        elif l.count(1) == 1:
            if l[0] == 1:
                A = np.array([
                    [2, 0, 1],
                    [0, 18, 0],
                    [1, 0, 0]
                ])
                b = np.array([12, 36, 4])
                try:
                    x = np.linalg.solve(A, b)
                    solve[0], solve[1], solve[2] = float(x[0]), float(x[1]), float(x[2])
                    all_solutions.append(solve)
                except np.linalg.LinAlgError:
                    all_solutions.append(None)
            elif l[1] == 1:
                A = np.array([
                    [2, 0, -1],
                    [0, 18, 0],
                    [-1, 0, 0]
                ])
                b = np.array([12, 36, 1])
                try:
                    x = np.linalg.solve(A, b)
                    solve[0], solve[1], solve[3] = float(x[0]), float(x[1]), float(x[2])
                    all_solutions.append(solve)
                except np.linalg.LinAlgError:
                    all_solutions.append(None)
            elif l[2] == 1:
                A = np.array([
                    [2, 0, 0],
                    [0, 18, 1],
                    [0, 1, 0]
                ])
                b = np.array([12, 36, 2])
                try:
                    x = np.linalg.solve(A, b)
                    solve[0], solve[1], solve[4] = float(x[0]), float(x[1]), float(x[2])
                    all_solutions.append(solve)
                except np.linalg.LinAlgError:
                    all_solutions.append(None)
            elif l[3] == 1:
                A = np.array([
                    [2, 0, 0],
                    [0, 18, -1],
                    [0, -1, 0]
                ])
                b = np.array([12, 36, -1])
                try:
                    x = np.linalg.solve(A, b)
                    solve[0], solve[1], solve[5] = float(x[0]), float(x[1]), float(x[2])
                    all_solutions.append(solve)
                except np.linalg.LinAlgError:
                    all_solutions.append(None)
        elif l.count(1) == 2:
            if l[0] == 1 and l[1] == 1:
                A = np.array([
                    [2, 0, 1, -1],
                    [0, 18, 0, 0],
                    [1, 0, 0, 0],
                    [-1, 0, 0, 0]
                ])
                b = np.array([12, 36, 4, 1])
                try:
                    x = np.linalg.solve(A, b)
                    solve[0], solve[1], solve[2], solve[3] = float(x[0]), float(x[1]), float(x[2]), float(x[3])
                    all_solutions.append(solve)
                except np.linalg.LinAlgError:
                    all_solutions.append(None)
            elif l[0] == 1 and l[2] == 1:
                A = np.array([
                    [2, 0, 1, 0],
                    [0, 18, 0, 1],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0]
                ])
                b = np.array([12, 36, 4, 2])
                try:
                    x = np.linalg.solve(A, b)
                    solve[0], solve[1], solve[2], solve[4] = float(x[0]), float(x[1]), float(x[2]), float(x[3])
                    all_solutions.append(solve)
                except np.linalg.LinAlgError:
                    all_solutions.append(None)
            elif l[0] == 1 and l[3] == 1:
                A = np.array([
                    [2, 0, 1, 0],
                    [0, 18, 0, -1],
                    [1, 0, 0, 0],
                    [0, -1, 0, 0]
                ])
                b = np.array([12, 36, 4, -2])
                try:
                    x = np.linalg.solve(A, b)
                    solve[0], solve[1], solve[2], solve[5] = float(x[0]), float(x[1]), float(x[2]), float(x[3])
                    all_solutions.append(solve)
                except np.linalg.LinAlgError:
                    all_solutions.append(None)
            elif l[1] == 1 and l[2] == 1:
                A = np.array([
                    [2, 0, -1, 0],
                    [0, 18, 0, 1],
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0]
                ])
                b = np.array([12, 36, 1, 2])
                try:
                    x = np.linalg.solve(A, b)
                    solve[0], solve[1], solve[3], solve[4] = float(x[0]), float(x[1]), float(x[2]), float(x[3])
                    all_solutions.append(solve)
                except np.linalg.LinAlgError:
                    all_solutions.append(None)
            elif l[1] == 1 and l[3] == 1:
                A = np.array([
                    [2, 0, -1, 0],
                    [0, 18, 0, -1],
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0]
                ])
                b = np.array([12, 36, 1, -1])
                try:
                    x = np.linalg.solve(A, b)
                    solve[0], solve[1], solve[3], solve[5] = float(x[0]), float(x[1]), float(x[2]), float(x[3])
                    all_solutions.append(solve)
                except np.linalg.LinAlgError:
                    all_solutions.append(None)
            elif l[2] == 1 and l[3] == 1:
                A = np.array([
                    [2, 0, 0, 0],
                    [0, 18, 1, -1],
                    [0, 1, 0, 0],
                    [0, -1, 0, 0]
                ])
                b = np.array([12, 36, 2, -1])
                try:
                    x = np.linalg.solve(A, b)
                    solve[0], solve[1], solve[4], solve[5] = float(x[0]), float(x[1]), float(x[2]), float(x[3])
                    all_solutions.append(solve)
                except np.linalg.LinAlgError:
                    all_solutions.append(None)
        elif l.count(1) == 3:
            if l[0] == 1 and l[1] == 1 and l[2] == 1:
                A = np.array([
                    [2, 0, 1, -1, 0],
                    [0, 18, 0, 0, 1],
                    [1, 0, 0, 0, 0],
                    [-1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0]
                ])
                b = np.array([12, 36, 4, 1, 2])
                try:
                    x = np.linalg.solve(A, b)
                    solve[0], solve[1], solve[2], solve[3], solve[4] = float(x[0]), float(x[1]), float(x[2]), float(
                        x[3]), float(x[4])
                    all_solutions.append(solve)
                except np.linalg.LinAlgError:
                    all_solutions.append(None)
            elif l[0] == 1 and l[1] == 1 and l[3] == 1:
                A = np.array([
                    [2, 0, 1, -1, 0],
                    [0, 18, 0, 0, -1],
                    [1, 0, 0, 0, 0],
                    [-1, 0, 0, 0, 0],
                    [0, -1, 0, 0, 0]
                ])
                b = np.array([12, 36, 4, 1, -1])
                try:
                    x = np.linalg.solve(A, b)
                    solve[0], solve[1], solve[2], solve[3], solve[5] = float(x[0]), float(x[1]), float(x[2]), float(
                        x[3]), float(x[4])
                    all_solutions.append(solve)
                except np.linalg.LinAlgError:
                    all_solutions.append(None)
            elif l[0] == 1 and l[2] == 1 and l[3] == 1:
                A = np.array([
                    [2, 0, 1, 0, 0],
                    [0, 18, 0, 1, -1],
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, -1, 0, 0, 0]
                ])
                b = np.array([12, 36, 4, 2, -1])
                try:
                    x = np.linalg.solve(A, b)
                    solve[0], solve[1], solve[2], solve[4], solve[5] = float(x[0]), float(x[1]), float(x[2]), float(
                        x[3]), float(x[4])
                    all_solutions.append(solve)
                except np.linalg.LinAlgError:
                    all_solutions.append(None)
            elif l[1] == 1 and l[2] == 1 and l[3] == 1:
                A = np.array([
                    [2, 0, -1, 0, 0],
                    [0, 18, 0, 1, -1],
                    [-1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, -1, 0, 0, 0]
                ])
                b = np.array([12, 36, 1, 2, -1])
                try:
                    x = np.linalg.solve(A, b)
                    solve[0], solve[1], solve[3], solve[4], solve[5] = float(x[0]), float(x[1]), float(x[2]), float(
                        x[3]), float(x[4])
                    all_solutions.append(solve)
                except np.linalg.LinAlgError:
                    all_solutions.append(None)
        else:
            A = np.array([
                [2, 0, 1, -1, 0, 0],
                [0, 18, 0, 0, 1, -1],
                [1, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, 0]
            ])
            b = np.array([12, 36, 4, 1, 2, -1])
            try:
                x = np.linalg.solve(A, b)
                solve[0], solve[1], solve[2], solve[3], solve[4], solve[5] = float(x[0]), float(x[1]), float(
                    x[2]), float(
                    x[3]), float(x[4]), float(x[5])
                all_solutions.append(solve)
            except np.linalg.LinAlgError:
                all_solutions.append(None)

    right_sol = []

    for sol in all_solutions:
        if sol is not None:
            if ogran(*sol):
                right_sol.append(sol)
    return right_sol


def ext_penalty_method(func, g1, g2, g3, g4):
    # функция штрафа
    def H(x1, x2):
        p = 0
        p += max(0, g1(x1, x2)) ** 2
        p += max(0, g2(x1, x2)) ** 2
        p += max(0, g3(x1, x2)) ** 2
        p += max(0, g4(x1, x2)) ** 2
        return p

    # функция, которую будем минимизировать
    def phi(x1, x2, r):
        return func(x1, x2) + r * H(x1, x2)

    r = 1
    eps = 10 ** (-5)
    x_curr = np.array([0, 1.5])  # x0

    while True:
        x_opt = minimize(lambda x: phi(x[0], x[1], r), x_curr, method='BFGS').x

        def curr_phi(x1, x2):
            return phi(x1, x2, r)

        x_0pt = coord_descent(x_curr, eps, curr_phi)[0]
        h_value = H(x_opt[0], x_opt[1])

        if h_value < eps:
            return x_opt

        # Увеличение штрафа и обновление точки
        r *= 10
        x_curr = x_opt
