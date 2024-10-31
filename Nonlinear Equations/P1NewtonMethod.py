import numpy as np

def f(x):
    return np.exp(np.sqrt(5) * x) - 13.5 * np.cos(0.1 * x) + 25 * x**4

def df(x):
    return np.sqrt(5) * np.exp(np.sqrt(5) * x) + 1.35 * np.sin(0.1 * x) + 100 * x**3

def newtons_method(f, df, x0, epsilon, max_iter):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(fx) < epsilon:
            return x
        
        x = x - fx / dfx
root1 = newtons_method(f, df, 1,1e-5,100) 
root2 = newtons_method(f, df, -1,1e-5,100) 
print(f"Root1: {root1}")
print(f"Root2: {root2}")