import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(np.sqrt(5) * x) - 13.5 * np.cos(0.1 * x) + 25 * x**4

def df(x):
    return np.sqrt(5) * np.exp(np.sqrt(5) * x) + 1.35 * np.sin(0.1 * x) + 100 * x**3

# Linear interpolation 
def linear_interpolation(f, x0, x1, epsilon, max_iter):
    ArrayLin = [f(x0)]
    for i in range(max_iter):
        f0, f1 = f(x0), f(x1)
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        ArrayLin.append(f(x2))
        if abs(x2 - x1) < epsilon:
            return x2, ArrayLin
        x0, x1 = x1, x2
    

# Newton
def newtons_method(f, df, x0, epsilon, max_iter):
    ArrayNewton = [f(x0)]
    x = x0
    for i in range(max_iter):
        fx, dfx = f(x), df(x)
        if abs(fx) < epsilon:
            return x, ArrayNewton
        x = x - fx / dfx
        ArrayNewton.append(f(x))

root_linear, Array_linear = linear_interpolation(f, 0, 1,1e-12,100)
root_newton, Array_newton = newtons_method(f, df, 1,1e-12,100)

plt.plot(Array_linear, label='Linear Interpolation', marker='o')
plt.plot(Array_newton, label="Newton Method", marker='x')
plt.plot(root_linear)
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.legend()
plt.show()
