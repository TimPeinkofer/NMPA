import numpy as np
import matplotlib.pyplot as plt

# Lane-Emden equation
def Lane_Emden_system(y, xi, n):
    theta, dtheta_dxi = y
    if xi == 0:
        return np.array([dtheta_dxi, 0])  # If we divide by zero
    return np.array([dtheta_dxi, - (2 / xi) * dtheta_dxi - theta ** n])

# Runge-Kutta 4 (RK4)
def rk4(System, y0, xi_max, h, n):
    xi_values = np.arange(1E-4, xi_max, h)  
    y = np.zeros((len(xi_values), len(y0)))
    y[0] = y0
    
    for i in range(1, len(xi_values)):
        xi = xi_values[i - 1]
        k1 = System(y[i - 1], xi, n)
        k2 = System(y[i - 1] + h * k1 / 2, xi + h / 2, n)
        k3 = System(y[i - 1] + h * k2 / 2, xi + h / 2, n)
        k4 = System(y[i - 1] + h * k3, xi + h, n)
        y[i] = y[i - 1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    return xi_values, y[:, 0], y[:, 1]  

# BC
y0 = np.array([1, 0])  
xi_max = 10
h = 0.001  

for n in [0, 1, 2, 3, 4]:
    xi_vals, theta_vals, dtheta_vals = rk4(Lane_Emden_system, y0, xi_max, h, n)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(xi_vals, theta_vals, label=f'Lane-Emden Solution (n={n})')
    plt.xlabel("Radius")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Lane-Emden equation with RK4 (n={n})")
    plt.grid()
    
   
    plt.savefig(f"Lane_Emden_n{n}.png", dpi=300)
    plt.close()  
