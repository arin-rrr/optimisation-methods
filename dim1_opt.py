import numpy as np
import matplotlib.pyplot as plt
from methods_opt0 import passive_search, dichotomy, golden_ratio, fibonacci_method
from method_opt1 import method_kasat, method_hord
from method_opt2 import newton_rafson

'''
Методы оптимизации одномерных функций
'''


def my_func(x):
    return x * np.sin(x) + 2 * np.cos(x)


def df(x):
    return -np.sin(x) + x * np.cos(x)


def df2(x):
    return -x * np.sin(x)


a, b = -5, -4
eps, delta = 0.02, 0.03

x = np.arange(a, b, 0.001)
fig, axes = plt.subplots(2, 1, figsize=(7, 8))

axes[0].plot(x, my_func(x), label='График функции')
axes[0].set_title("Моя функция")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(x, df(x), label='График производной')
axes[1].set_title("График производной")
axes[1].set_xlabel("x")
axes[1].set_ylabel("df(x)")
axes[1].grid(True)
axes[1].legend()

print("Отрезок [-5, -4], погрешность = 0.02")
print()
print("Метод пассивного поиска")
print(f"x_min = {passive_search(a, b, eps, my_func)[0]}, погрешность = {eps}")
print(f"f(x_min) = {passive_search(a, b, eps, my_func)[1]}")
print(f"Количество обращений к правой части = {passive_search(a, b, eps, my_func)[2]}")
print()
print("Метод дихотомии")
print(f"x_min = {dichotomy(a, b, eps, delta, my_func)[0]}, погрешность = {eps}")
print(f"f(x_min) = {dichotomy(a, b, eps, delta, my_func)[1]}")
print(f"Количество обращений к правой части = {dichotomy(a, b, eps, delta, my_func)[2]}")
print()
print("Метод золотого сечения")
print(f"x_min = {golden_ratio(a, b, eps, my_func)[0]}, погрешность = {eps}")
print(f"f(x_min) = {golden_ratio(a, b, eps, my_func)[1]}")
print(f"Количество обращений к правой части = {golden_ratio(a, b, eps, my_func)[2]}")
print()
print("Метод Фибоначчи")
print(f"x_min = {fibonacci_method(a, b, eps, my_func)[0]}, погрешность = {eps}")
print(f"f(x_min) = {fibonacci_method(a, b, eps, my_func)[1]}")
print(f"Количество обращений к правой части = {fibonacci_method(a, b, eps, my_func)[2]}")
print()
print("Метод касательных")
print('По графику - функция выпуклая и непр.дифф')
print(f"Проверим, что f'(a) < 0 и f'(b) > 0, иначе a или b - минимум: f'(a) = {df(a)}, f'(b) = {df(b)}")
print(f"x_min = {method_kasat(a, b, eps, my_func, df)[0]}, погрешность = {eps}")
print(f"f(x_min) = {method_kasat(a, b, eps, my_func, df)[1]}")
print(f"Количество обращений к правой части функции = {method_kasat(a, b, eps, my_func, df)[2]}")
print(f"Количество обращений к производной функции = {method_kasat(a, b, eps, my_func, df)[3]}")
print()
print("Метод Ньютона-Рафсона")
print(f"x_min = {newton_rafson(a, eps, my_func, df, df2)[0]}, погрешность = {eps}")
print(f"f(x_min) = {newton_rafson(a, eps, my_func, df, df2)[1]}")
print(f"Количество обращений к производной функции = {newton_rafson(a, eps, my_func, df, df2)[2]}")
print(f"Количество обращений ко второй производной функции = {newton_rafson(a, eps, my_func, df, df2)[3]}")
print()
print("Метод хорд (секущих)")
print(f"x_min = {method_hord(a, b, eps, my_func, df)[0]}, погрешность = {eps}")
print(f"f(x_min) = {method_hord(a, b, eps, my_func, df)[1]}")
print(f"Количество обращений к производной функции = {method_hord(a, b, eps, my_func, df)[2]}")

# на графике отобразим все найденные минимумы и реальный минимум
x_min = -4.49341
y_min = -4.82057

x_min_pass = passive_search(a, b, eps, my_func)[0]
y_min_pass = passive_search(a, b, eps, my_func)[1]

x_min_dic = dichotomy(a, b, eps, delta, my_func)[0]
y_min_dic = dichotomy(a, b, eps, delta, my_func)[1]

x_min_gold = golden_ratio(a, b, eps, my_func)[0]
y_min_gold = golden_ratio(a, b, eps, my_func)[1]

x_min_fib = fibonacci_method(a, b, eps, my_func)[0]
y_min_fib = fibonacci_method(a, b, eps, my_func)[1]

x_min_kas = method_kasat(a, b, eps, my_func, df)[0]
y_min_kas = method_kasat(a, b, eps, my_func, df)[1]

x_min_new = newton_rafson(a, eps, my_func, df, df2)[0]
y_min_new = newton_rafson(a, eps, my_func, df, df2)[1]

x_min_hord = method_hord(a, b, eps, my_func, df)[0]
y_min_hord = method_hord(a, b, eps, my_func, df)[1]

axes[0].scatter(x_min, y_min, s=20, c='red')
axes[0].scatter(x_min_pass, y_min_pass, s=10, c='red', alpha=0.5)
axes[0].scatter(x_min_dic, y_min_dic, s=10, c='green', alpha=0.5)
axes[0].scatter(x_min_gold, y_min_gold, s=10, c='blue', alpha=0.5)
axes[0].scatter(x_min_fib, y_min_fib, s=10, c='orange', alpha=0.5)
axes[0].scatter(x_min_kas, y_min_kas, s=10, c='purple', alpha=0.5)
axes[0].scatter(x_min_new, y_min_new, s=10, c='pink', alpha=0.5)
axes[0].scatter(x_min_hord, y_min_hord, s=10, c='cyan', alpha=0.5)

plt.tight_layout()
plt.show()
