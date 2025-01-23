import numpy as np
import matplotlib.pyplot as plt

def prey(x, y):
    return 0.1 * x - 0.02 * x * y

def predator(x, y):
    return 0.01 * x * y - 0.1 * y

def runge_kutta_4th(x0, y0, dt, t_end):
    t = np.arange(0, t_end + dt, dt)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    
    # Initial conditions
    x[0] = x0
    y[0] = y0

    for i in range(1, len(t)):
        # k1 values
        k1 = prey(x[i-1], y[i-1]) * dt
        l1 = predator(x[i-1], y[i-1]) * dt

        # k2 values
        k2 = prey(x[i-1] + 0.5 * k1, y[i-1] + 0.5 * l1) * dt
        l2 = predator(x[i-1] + 0.5 * k1, y[i-1] + 0.5 * l1) * dt

        # k3 values
        k3 = prey(x[i-1] + 0.5 * k2, y[i-1] + 0.5 * l2) * dt
        l3 = predator(x[i-1] + 0.5 * k2, y[i-1] + 0.5 * l2) * dt

        # k4 values
        k4 = prey(x[i-1] + k3, y[i-1] + l3) * dt
        l4 = predator(x[i-1] + k3, y[i-1] + l3) * dt

        # Update x and y
        x[i] = x[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y[i] = y[i-1] + (l1 + 2 * l2 + 2 * l3 + l4) / 6

    return t, x, y


#Initial values
x0 = 40       
y0 = 9        
dt = 0.01     
t_end = 300   


t, x, y = runge_kutta_4th(x0, y0, dt, t_end)


plt.figure(figsize=(16, 9))
plt.plot(t, x, label="Prey Population (x)")
plt.plot(t, y, label="Predator Population (y)")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Predator-Prey ODE")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(x, y)
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")
plt.title("Phase Diagramm Predator-Prey")
plt.legend()
plt.grid()
plt.show()
