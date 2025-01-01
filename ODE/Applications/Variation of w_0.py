import numpy as np
import matplotlib.pyplot as plt
b = 0.707
a = 2

# Definition of the function
def f(x, y):
    return -a * y + b**2 * y**3

# Runge-Kutta 4th order
def runge_kutta(f, y_0, x_start, x_end, n):
    h = (x_end - x_start) / n  
    x_values = np.linspace(x_start, x_end, n + 1)
    y_values = [y_0]  
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

# Parameter
n = 1000
x_start = 0
x_end = 10  
y_0 = 2  
w_0 = 1/np.sqrt(2)
w_0_values = np.linspace(w_0-0.5, w_0, 7)
w_0_values_2 = np.linspace(w_0, w_0+0.5, 7)


plt.figure(figsize=(16, 9))

#Plot for w_0 < 0.707
for i in w_0_values:
    b = i
    x, res = runge_kutta(f, y_0, x_start, x_end, n)
    plt.plot(x, res, label="b = {}".format(i.round(3)))

plt.title("Numerical Solution der DGL $y' = -ay + b^2y^3$")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(16, 9))

#Plot for w_0 > 0.707
for i in w_0_values_2:
    b = i
    x, res = runge_kutta(f, y_0, x_start, x_end, n)
    plt.plot(x, res, label="b = {}".format(i.round(3)))

plt.title("Numerical Solution der DGL $y' = -ay + b^2y^3$")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()




