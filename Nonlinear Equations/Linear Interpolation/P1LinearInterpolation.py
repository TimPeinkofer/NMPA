import numpy as np

def f(x):
    return np.exp(np.sqrt(5) * x) - 13.5 * np.cos(0.1 * x) + 25 * x**4

def linear_interpolation(f, x0, x1, epsilon, max_iter):
    for i in range(max_iter):
        f0, f1 = f(x0), f(x1)

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        
        if abs(x2 - x1) < epsilon:
            print(f(x2))
            return x2
        x0, x1 = x1, x2

root1 = linear_interpolation(f, 0, 1,1e-5,100) 
root2 = linear_interpolation(f, 0, -1,1e-5,100) 
print(f"Solution: {root1}")
print(f"Solution: {root2}")
