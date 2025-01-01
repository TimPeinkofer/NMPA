import numpy as np
import matplotlib.pyplot as plt

#defintion of the function
def f(x,y,alpha=0):
    b = x**2-2*x+np.sin(x)
    a = 1
    return b*y**alpha+ a*y


#Heun's method for the first steps
def heun_method(f, y_0, x_0, h):
    k1 = f(x_0, y_0)
    k2 = f(x_0 + h, y_0 + h * k1)
    return y_0 + (h / 2) * (k1 + k2)

def adam_predictor(f, y_0, x, n, h): #Predictor method
    y_values = [y_0]  # Solution values list
    
    for i in range(3): #Calculate the first values via Heun method
        y_values.append(heun_method(f, y_values[i], x[i], h))
    
    for i in range(3, n - 1): #Calculte the rest via Adams method
        y = y_values[i] + (h / 24) * ( 55 * f(x[i], y_values[i]) - 59 * f(x[i-1], y_values[i-1]) 
                                      +37* f(x[i-2], y_values[i-2]) -9*f(x[i-3], y_values[i-3]))
        y_values.append(y)  

    return np.array(y_values)

def adams_corrector(f, y_values, x, n, h): #Corrector method
    
    for i in range(3, n - 1):
        y_corr = y_values[i] + (h / 24) * (
            9 * f(x[i + 1], y_values[i + 1]) +
            19 * f(x[i], y_values[i]) -
            5 * f(x[i - 1], y_values[i - 1]) +
            f(x[i - 2], y_values[i - 2])
        )
        y_values[i + 1] = y_corr

    return np.array(y_values)

def adams_ode_int(f, y_0, x, n, h): #Combination of both methods
    y_pred = adam_predictor(f, y_0, x, n, h)
    y_corr = adams_corrector(f, y_pred, x, n, h)
    return y_corr, y_pred


x = np.linspace(0, 2, 1000)
y_corr, y_pred = adams_ode_int(f, 0.1, x, 1000, 2/1000)

#Plot
plt.figure(figsize=(16, 9))
plt.plot(x, y_pred, color='blue', label='Adams method without corrector')
plt.plot(x, y_corr, color='red', linestyle='--', label='Adams method with corrector')
plt.title("Numerical solution of the ODE $y' = x^2-2x+sin(x)+y$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
