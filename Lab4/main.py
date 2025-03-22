from math import log, exp
from scipy.optimize import minimize

"""20. Метанол + этилацетат"""

T = 298.15
R = 8.314

# VLE
x1 = [0.1, 0.3, 0.5, 0.7, 0.9]
y1 = [0.276, 0.454, 0.579, 0.64, 0.828]
P = [0.1615, 0.1887, 0.1977, 0.1982, 0.1860]

n = len(x1)

# Params
A = 18.59, 16.15
B = 3626.55, 2790.50
C = -34.29, -57.15

Tc = 512.60, 523.20
Pc = 80.96, 38.30
W = 0.56, 0.36


def antoine_equation(a, b, c, t):
    return exp(a - b / (t + c)) / 750.062


# def calculate_molar_volumes(tc, pc, w):
#     z_ra = lambda omega: 0.29056 - 0.08775 * omega
#     return R * tc / pc * z_ra(w)


def func(params):
    global g_exp, x1, x2
    lambda12, lambda21 = params
    res = 0.0

    for j in range(len(x1)):
        # Расчет избыточной энергии Гиббса по модели Вильсона
        g_calc = -R * T * (x1[j] * log(x1[j] + lambda12 * x2[j]) + x2[j] * log(x2[j] + lambda21 * x1[j]))
        res += abs(g_calc - g_exp[j])  # Сумма абсолютных отклонений
    return res


if __name__ == '__main__':
    # 1. Рассчитать по уравнению Антуана
    P1_0, P2_0 = antoine_equation(A[0], B[0], C[0], T), antoine_equation(A[1], B[1], C[1], T)

    # 2. Выразить из основного уравнения подхода
    gamma1, gamma2 = [], []
    x2, y2 = [], []

    for i in range(len(x1)):
        x2.append(1 - x1[i])
        y2.append(1 - y1[i])

        gamma1.append(y1[i] * P[i] / (P1_0 * x1[i]))
        gamma2.append(y2[i] * P[i] / (P2_0 * x2[i]))

    # 3. Для каждой точки рассчитать экспериментальную избыточную мольную энергию Гиббса
    g_exp = []
    for i in range(len(x1)):
        g_exp.append(R * T * (x1[i] * log(gamma1[i]) + x2[i] * log(gamma2[i])))

    # 4. Подобрать такие Λ12 и Λ21, чтобы разница в избыточной энергии Гиббса между
    # эксп. и расчетной (Вильсон) была минимальна
    res = minimize(func, [0.1, 0.1], method='Nelder-Mead', tol=1e-6)

    print(res.message)
    # print(res.x)

    for i in range(len(x1)):
        print(res.x * x1[i] * x2[i])

    # Сделать график