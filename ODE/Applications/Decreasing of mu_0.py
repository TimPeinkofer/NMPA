import numpy as np
import matplotlib.pyplot as plt

# Definition of the function
def f(x, y, a):
    b = (1 / np.sqrt(2)) * np.tan(x)
    return -a * y + b**2 * y**3

# Runge-Kutta 4th order
def runge_kutta(f, y_0, x_start, x_end, n, a):
    h = (x_end - x_start) / n  
    x_values = np.linspace(x_start, x_end, n + 1)
    y_values = [y_0]  
    y = y_0

    for i in range(n):
        x_i = x_values[i]
        k_1 = h * f(x_i, y, a)
        k_2 = h * f(x_i + 0.5 * h, y + 0.5 * k_1, a)
        k_3 = h * f(x_i + 0.5 * h, y + 0.5 * k_2, a)
        k_4 = h * f(x_i + h, y + k_3, a)

        y = y + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        y_values.append(y)

    return x_values, np.array(y_values)

# Parameter
n = 1000
x_start = 0
x_end = 10  
y_0 = 2  
mu_0_values = np.linspace(2, 1.5, 20)  

plt.figure(figsize=(16, 9))

# Plot for different mu_0 values
for mu_0 in mu_0_values:
    x, res = runge_kutta(f, y_0, x_start, x_end, n, mu_0)
    plt.plot(x, res, label="mu_0 = {:.3f}".format(mu_0))

plt.title("Numerical solution of the DGL $y' = -ay + b^2y^3$")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid(True)
plt.legend()
plt.show()
