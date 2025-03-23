import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

"""20. Метанол + этилацетат"""

R = 8.314
T = 298.15

# VLE
x1 = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
y1 = np.array([0.276, 0.454, 0.579, 0.64, 0.828])
P = np.array([0.1615, 0.1887, 0.1977, 0.1982, 0.1860])

# Params
A1, B1, C1 = 18.59, 3626.55, -34.29
A2, B2, C2 = 16.15, 2790.50, -57.15


# Функция для расчета давления насыщения
def antoine_equation(a, b, c, t):
    return np.exp(a - b / (t + c)) / 750.062


# Функция для расчета коэффициентов активности
def calculate_activity_coefficients():
    g1 = y1 * P / (P1_0 * x1)
    g2 = y2 * P / (P2_0 * x2)
    g_exp = (R * T) * (x1 * np.log(g1) + x2 * np.log(g2))
    return g1, g2, g_exp


def wilson_func(params, x1, x2, g_exp):
    lam12, lam21 = params
    g_calc = (R * T) * (-x1 * np.log(x1 + lam12 * x2) - x2 * np.log(lam21 * x1 + x2))
    return np.sum(np.abs(g_exp - g_calc))


# Функция для построения графиков
def plot_diagrams():
    plt.figure(figsize=(9, 6))

    plt.plot(x1_range, p_total, label='Модель ассоциации')
    plt.plot(y1_calc, p_total, label='Модель ассоциации')
    plt.scatter(x1, P, color='red', label='Экспериментальные данные')
    plt.xlabel('X (метанол)')
    plt.ylabel('Давление (бар)')
    plt.title('P-xy диаграмма системы "метанол + этилацетат" при T = 298.15 K')
    plt.legend()
    plt.grid()

    # plt.subplot(1, 2, 2)
    # plt.plot(x1_range, y1_calc, label='Calculated y1')
    # plt.plot(x1_range, x1_range, label='x = y', color='black', linestyle='--')
    # plt.scatter(x1, y1, color='blue', label='Экспериментальные данные')
    # plt.xlabel('Mole Fraction of Acetone in Liquid (x1)')
    # plt.ylabel('Mole Fraction of Acetone in Vapor (y1)')
    # plt.title('y-x диаграмма системы "метанол + этилацетат" при T = 298.15 K')
    # plt.legend()
    # plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 1. Рассчитать по уравнению Антуана
    P1_0 = antoine_equation(A1, B1, C1, T)
    P2_0 = antoine_equation(A2, B2, C2, T)

    # 2. Выразить из основного уравнения подхода
    x2, y2 = 1 - x1, 1 - y1
    gam1, gam2, gE_exp = calculate_activity_coefficients()

    initial_guess = [0.1, 0.1]
    result = minimize(wilson_func, np.array(initial_guess), args=(x1, x2, gE_exp), method='Nelder-Mead', tol=1e-6)

    lambda12, lambda21 = result.x
    print("Оптимизированные параметры модели Вильсона:")
    print(f'{lambda12 = } {lambda21 = }')

    x1_range = np.linspace(0, 1, 100)
    x2_range = 1 - x1_range

    gamma1 = np.exp(-np.log(x1_range + lambda12 * x2_range) + x2_range * (
            lambda12 / (x1_range + lambda12 * x2_range) - lambda21 / (lambda21 * x1_range + x2_range)))
    gamma2 = np.exp(-np.log(x2_range + lambda21 * x1_range) - x1_range * (
            lambda12 / (x1_range + lambda12 * x2_range) - lambda21 / (lambda21 * x1_range + x2_range)))

    p_total = x1_range * gamma1 * P1_0 + x2_range * gamma2 * P2_0
    y1_calc = (x1_range * gamma1 * P1_0) / p_total
    y2_calc = (x2_range * gamma2 * P2_0) / p_total

    plot_diagrams()
