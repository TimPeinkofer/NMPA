import numpy as np
import sympy as sp

# Define variables
y = sp.Symbol('y')
x = sp.Symbol('x')

# Define function
def func(x):
    return x * y**2

# Define Simpson 1/3
def Simpson(h, values):  
    sum_result = values[0] + values[-1]  # Get the sum of the values of the integral limits

    for i in range(1, len(values)):  # Sum all other values based on the number of steps
        if i % 2 == 0:  # Multiply all odd index values with 2 and the others with 4 and calculate the sum
            sum_result += 4 * values[i]
        else:
            sum_result += 2 * values[i]

    result = h / 3 * sum_result  # Get the result
    return result


sol = {}


indices = [400, 600]


for i, index in enumerate(indices):
    h_1 = (2-2*y)/index #Define the steps for both integrals
    h_2 = 1/index

    x_vals = [2*y + j * h_1 for j in range(index+ 1)]

    # Calculate values
    f_x = [func(xi) for xi in x_vals]
    
    # Calculate integral 1
    f_y = Simpson(h_1, f_x)
    x_vals = [j * h_2 for j in range(index+ 1)]
    result_func = sp.lambdify(y, f_y, 'sympy')
    
    # Calculate values
    f = [result_func(xi) for xi in x_vals]
    sol[i] = Simpson(h_2, f)

err = np.abs(sol[1]-sol[0])


for i in sol:
    print(f"Ergebnis für {indices[i]} Intervalle: {sp.simplify(sol[i])}")
print(f"Error:", err)