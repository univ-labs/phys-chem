from scipy.optimize import minimize
import numpy as np
import math

T = 333.15
R = 8.314
x1 = [0.7, 0.6, 0.5, 0.4, 0.3]
y1 = [0.8, 0.5, 0.43, 0.32, 0.22]
P = [1, 2, 3, 4, 5]
arr1 = []

for i in range(0, 5):
    gam1[i] = y1[i] * P[i] / (P1_0 * x1[i])
    gam2[i] = (1 - y1[i]) * P[i] / (P2_0 * (1 - x1[i]))
    arr1.append(gam1[i] * x1[i] + gam2[i] * (1 - x1[i]))


def func(x, x1=x1, ge_exp=arr1):
    A = x[0]
    sum = 0.0
    for i in range(0, 5):
        sum += abs(arr1[i] - A * x1[i] * (1 - x1[i]))
    return sum


res = minimize(func, x0=[1.0], method='Nelder-Mead', tol=1e-6)
print(res.message)
print(res.x)
print(arr1)

for i in range(0, 5):
    print(arr1[i])
    print(res.x * x1[i] * (1 - x1[i]))
