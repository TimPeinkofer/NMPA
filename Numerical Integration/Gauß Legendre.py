import numpy as np

# Define Function 
def func(x):
    return 2**x*np.sin(x)/x

# Transformation for our used Intervall
def transform(x, b, a):
    return 0.5*(b - a) * x + 0.5*(b + a)

# Gauss-Legendre weights and nodes
A_i = [0.3478548451, 0.6521451549, 0.3478548451, 0.6521451549]
x_i = [0.8611363116, 0.3394810436, -0.8611363116, -0.3394810436]

# Gauß-Legendre function
def Gauss_legendre(b, a, x_v, A):
    sum = 0
    x_transformed = [transform(r, b, a) for r in x_v]  # Transform nodes for use
    
    for i in range(len(x_transformed)):
        sum += A[i] * func(x_transformed[i])  # Calculate the values with the weigths and nodes

    return 0.5 * (b - a) * sum  # Multiply by the scaling factor



result = Gauss_legendre(np.pi, 1, x_i, A_i) #Calculate the integral

new_int = (np.pi+1) / 2 
result_2 = Gauss_legendre(new_int, 1, x_i, A_i) + Gauss_legendre(np.pi, new_int, x_i, A_i) #Calculate the integral with finer steps for error calculation
    
err = abs(result_2 - result) #Calculate the error

print("Solution of the integral:\n ", result)
print("Error:\n ", err)