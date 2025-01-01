import numpy as np
import matplotlib.pyplot as plt

# Definition of the function
def f(x,y,alpha=0):
    b = x**2-2*x+np.sin(x)
    a = 1
    return b*y**alpha+ a*y

# 4th Runge-Kutta-method 
def runge_kutta(f, y_0, x_start, x_end, n):
    h = (x_end - x_start) / n  # Stepsize
    x_values = np.linspace(x_start, x_end, n + 1)  
    y_values = [y_0]  # Solution values
    y = y_0

    for i in range(n):
        x_i = x_values[i]
        k_1 = h * f(x_i, y)
        k_2 = h * f(x_i + 0.5 * h, y + 0.5 * k_1)
        k_3 = h * f(x_i + 0.5 * h, y + 0.5 * k_2)
        k_4 = h * f(x_i + h, y + k_3)

        y = y + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        y_values.append(y)  

    return x_values, np.array(y_values)

# Values for calculation
n = 1000
x_start = 0
x_end = 2
y_0 = 0.1


x, res = runge_kutta(f, y_0, x_start, x_end, n)

# Plot
plt.figure(figsize=(16, 9))
plt.plot(x, res, label="4th order Runge Kutta method")
plt.title("Numerical Solution of the ODE $y' = y + x^2-2x+sin(x)$")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()
