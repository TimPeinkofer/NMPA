import numpy as np


def Newton_cotes(n,x, f):

    h = (x[-1] - x[0]) / (n - 1)  # Calculating the stepsize h
    sum_integral = 0

    for i in range(0, n - 2, 2):  # Calculating the value for every odd step
        sum_1 = f[i] + 4 * f[i + 1] + f[i + 2]
        sum_integral += h/3 * sum_1

    return sum_integral


# Testfunction
def func(x):
    return x

n_1 = 1000
n_2 = 2000

x = np.linspace(0, 2, n_1)  
f = [func(x_i) for x_i in x]  


x_2 = np.linspace(0, 2, n_2)
f_2 = [func(x_i) for x_i in x_2]

# Integration for two different n to get the error of our integration
result_1 = Newton_cotes(n_1,x, f)  
result_2 = Newton_cotes(n_2,x_2, f_2) 
err = np.abs(result_1 - result_2)

# Ergebnisse ausgeben
print(f"Result of our Integration: {result_1}")
print(f"Error of our Integration: {err}")
