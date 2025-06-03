import numpy as np
import matplotlib.pyplot as plt
from cond_opt_methods import Kuhn_Tucker, ext_penalty_method

'''
Условная оптимизация
'''


def my_func(x1, x2):
    return x1 ** 2 + 9 * x2 ** 2 - 12 * x1 - 36 * x2


def ogran(x1, x2, l1, l2, l3, l4):
    return -1 <= x1 <= 4 and 1 <= x2 <= 2 and l1 >= 0 and l2 >= 0 and l3 >= 0 and l4 >= 0


def g1(x1, x2):
    return x1 - 4


def g2(x1, x2):
    return -1 - x1


def g3(x1, x2):
    return x2 - 2


def g4(x1, x2):
    return 1 - x2


# строим функцию в области ограничений
x = np.linspace(-1, 4, 200)
y = np.linspace(1, 2, 200)
X, Y = np.meshgrid(x, y)
Z = my_func(X, Y)

# Создание 3D-осей
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Построение поверхности
ax.plot_surface(X, Y, Z, cmap='viridis')

sol = Kuhn_Tucker(my_func, ogran)
print("Метод Куна-Такера")
print(f"Подходящие решения: {sol}")
print(f"Минимум функции {my_func(sol[0][0], sol[0][1])}")
print()
print('Метод внешних штрафов')
sol_ext = ext_penalty_method(my_func, g1, g2, g3, g4)
print(f'Решение: {sol_ext}')
print(f"Минимум функции: {my_func(sol_ext[0], sol_ext[1])}")
plt.show()
