import numpy as np 
import matplotlib.pyplot as plt

# Constants
L = 2
w_1 = 3
w_2 = 4.5

# Define the wavefunction
def wave_func(x, t):
    t_1 = 1 / np.sqrt(L) * np.sin(np.pi * x / L) * np.exp(-1j * w_1 * t)
    t_2 = 1 / np.sqrt(L) * np.sin(2 * np.pi * x / L) * np.exp(-1j * w_2 * t)
    return t_1 + t_2 if x < L else 0

# Define the probability density
def probability(x, t):
    return np.abs(wave_func(x, t)) ** 2

# Filling an array with the analytical function
def array_filling(h, a, b, func, array, n, t):
    if n != len(array):
        raise ValueError("Please choose matching values for calculation")  # Compare the size of the array and the number of data points

    for i in range(n):  # Calculate the value for every data point until the upper bound is reached
        x = a + i * h
        if x > b:
            break
        array[i] = func(x, t)

    return array

# Newton-Cotes function
def Newton_cotes(n, x, f):
    h = (x[-1] - x[0]) / (n - 1)  # Calculating the step size h
    sum_integral = 0

    for i in range(0, n - 2, 2):  # Calculating the value for every odd step
        sum_1 = f[i] + 4 * f[i + 1] + f[i + 2]
        sum_integral += h / 3 * sum_1

    return sum_integral

interval_a, interval_b = 3 * L / 4, L  # Bounds
time_steps = [0, 2 * np.pi / (w_2 - w_1)]  # Set time values

for t in time_steps:# Calculate the values for every time 

    results = [] 
    arr_c = [0] * 10000
    h = (interval_b - interval_a) / (10000 - 1)
    check_arr = array_filling(h, interval_a, interval_b, probability, arr_c, 10000, t) #Calculate comparison values for error calculation
    check_int = Newton_cotes(10000, arr_c, check_arr)
    print(f"t = {t}")
    print("\n")

    for n in range(5, 501, 2):  # Calculate values for every odd n
        num_points = n  
        h = (interval_b - interval_a) / (num_points - 1)  # Compute step size h
        arr = [0] * num_points  

        # Fill array with analytical function
        array_filled = array_filling(h, interval_a, interval_b, probability, arr, n, t)

        # Integrate using Newton Cotes 
        integral_result = Newton_cotes(n, arr, array_filled)
        error = np.abs(integral_result - check_int)
        
        results.append((n, np.log(h), np.log(error)))
        print(f"n: {n}, log(h): {np.log(h):.5f}, log(E): {np.log(error):.5f}")
    
    print("\n")
    
    result_array = np.array(results) #Plot results for log(h) and log(E)
    plt.figure(figsize=(10, 6))
    plt.plot(result_array[:,1], result_array[:,2])
    plt.xlabel("log(h)")
    plt.ylabel("log(E)")
    plt.title(f"Convergence of Newton-Cotes 2nd order for t = {t:.5f} s")
    plt.grid()
    plt.show()
