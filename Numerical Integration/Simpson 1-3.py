import numpy as np


def func(x): # Define function
    return 2**x*np.sin(x)/x



def Simpson(h,values,x): #Simpson Rule 1/3
    sum = 0
    sum = values[0]+values[-1] # Get the sum of the values of the integral limits

    for i in range(1,len(x)): # Sum all other values based on the number of steps
        
        if i % 2 == 0: # Multiply all odd index values with 2 and the others with 4 and calculate the sum
            sum += 4*values[i]
        
        else:
            sum += 2*values[i]
    
    result = h/3*sum # Get the result

    return result

sol = {}


indices = [300, 600]


for i, index in enumerate(indices):
    x = np.linspace(1, 2, index)  
    f_x = [func(x_i) for x_i in x]  
    h = (x[-1] - x[0]) / (len(x) - 1)  # Calculate step size for our iteration
    sol[i] = Simpson(h, f_x, x)  



err = np.abs(sol[1]-sol[0]) #Calculate the error

print("Solution of the integral:\n ", sol[0])
print("Error:\n ", err)
