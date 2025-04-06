import numpy as np
import matplotlib.pyplot as plt
from methods_dim2 import coord_descent, grad_descent_step_split, mngs, fast_grad_p, ovr_method


def my_func2(x1, x2):
    return x1 + 2 * x2 + 4 * (1 + x1 ** 2 + x2 ** 2) ** 0.5


def df2(x1, x2):
    return [1 + 4 * x1 / (1 + x1 ** 2 + x2 ** 2) ** 0.5, 2 + 4 * x2 / (1 + x1 ** 2 + x2 ** 2) ** 0.5]


x = np.linspace(-7, 7, 200)
y = np.linspace(-7, 7, 200)
X, Y = np.meshgrid(x, y)
Z = my_func2(X, Y)

# Создание 3D-осей
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Построение поверхности
ax.plot_surface(X, Y, Z, cmap='viridis')

x0 = [0, 0]
eps = 0.02

x1_min_desc = coord_descent(x0, eps, my_func2)[0]
y1_min_desc = coord_descent(x0, eps, my_func2)[1]

print()
print("Метод покоординатного спуска")
print(f"x_min = {x1_min_desc}, погрешность = {eps}")
print(f"f(x_min) = {y1_min_desc}")
print()

x1_min_split_step = grad_descent_step_split(x0, my_func2, df2)
print("Градиентный спуск с дроблением шага")
print(f"x_min = {x1_min_split_step}")
print(f"f(x_min) = {my_func2(x1_min_split_step[0], x1_min_split_step[1])}")

print()
x1_min_mngs = mngs(x0, my_func2, df2)
print("МНГС")
print(f"x_min={x1_min_mngs}")
print(f"f(x_min) = {my_func2(x1_min_mngs[0], x1_min_mngs[1])}")

x1_min_p = fast_grad_p(x0, my_func2, df2)
print()
print("Ускоренный градиентный метод p-ого порядка")
print(f"x_min={x1_min_p}")
print(f"f(x_min) = {my_func2(x1_min_p[0], x1_min_p[1])}")

x_min_ovr = ovr_method(x0, my_func2, df2)
print()
print("Овражный метод")
print(f"x_min={x_min_ovr}")
print(f"f(x_min) = {my_func2(x_min_ovr[0], x_min_ovr[1])}")
plt.show()
