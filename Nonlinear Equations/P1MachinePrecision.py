import numpy as np
#Newton
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
            return x,i
        x = x - fx / dfx
    return x,max_iter
#linear Interpolation
def linear_interpolation(f, x0, x1, epsilon, max_iter):
    for i in range(max_iter):
        f0, f1 = f(x0), f(x1)

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        
        if abs(x2 - x1) < epsilon:
            print(f(x2))
            return x2,i
        x0, x1 = x1, x2
    return x2,max_iter


rootNewton1,Iteration_Newton1 = newtons_method(f, df, 1,2.22e-5,1000) 
rootNewton2,Iteration_Newton2 = newtons_method(f, df, -1,2.22e-5,1000)
rootLin1,Iteration_Lin1 = linear_interpolation(f, 0, 1,2.22e-5,1000)
rootLin2,Iteration_Lin2 = linear_interpolation(f, -1, 0,2.22e-5,1000)
print(f"IterationNewton1: {Iteration_Newton1}")
print(f"IterationNewton2: {Iteration_Newton2}") 
print(f"IterationLin1: {Iteration_Lin1}")
print(f"IterationLin2: {Iteration_Lin2}")