import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.cos(x) + x*np.sin(x)

def df(x):
    return x * np.cos(x)

def newtons_method(f, df, guess, epsilon=1e-5, max_iter=100):
    Lambda = guess
    for i in range(max_iter):
        f_val = f(Lambda)
        df_val = df(Lambda)
        if abs(f_val) < epsilon:  
            return Lambda
        
        Lambda = Lambda - f_val / df_val
    return Lambda 

initial_guesses = [3,6,9,12]

eigenvalues_newton = [newtons_method(f, df, guess) for guess in initial_guesses]

print("Eigenvalues:")
for eigenvalues in eigenvalues_newton:
    print(eigenvalues)