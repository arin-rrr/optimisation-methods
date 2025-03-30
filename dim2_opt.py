import numpy as np
import matplotlib.pyplot as plt
from methods_dim2 import coord_descent


def my_func2(x1, x2):
    return x1 + 2 * x2 + 4 * (1 + x1 ** 2 + x2 ** 2) ** 0.5

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


plt.show()