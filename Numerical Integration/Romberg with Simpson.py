import numpy as np

# Define function
def func(x):
    return 2**x * np.sin(x) / x



# Simpson 1/3 Rule
def Simpson(h, values):
    sum = values[0] + values[-1]
    
    # Apply Simpson's rule for odd and even indices
    for i in range(1, len(values) - 1): # Multiply values with to if they ar odd, else with four and calculate sum
        if i % 2 == 0:
            sum += 4 * values[i]  
        else:
            sum += 2 * values[i]  
    
    result = h / 3 * sum  # Calculate result
    return result

# Romberg Integration
def Romberg(I_1, I_2, n=4): # Error h^4 because we use Simpson 1/3
    return I_2 + (I_2 - I_1) / (2**n - 1)


indices = [300,600, 400, 800]
sol = {}

for i, index in enumerate(indices): # Calculate the result for different steps
    x = np.linspace(1, np.pi, index)  
    f_x = [func(x_i) for x_i in x]
    h = (x[-1] - x[0]) / (len(x) - 1) 
    sol[i] = Simpson(h,f_x)



# Apply Romberg method for higher accuracy
result_1 = Romberg(sol[0], sol[1])
result_2 = Romberg(sol[0], sol[1])

err = np.abs(result_2-result_1)

print(f"Final result:",result_1)
print(f"Error:",err) # Must be zero, just for own knowledge
