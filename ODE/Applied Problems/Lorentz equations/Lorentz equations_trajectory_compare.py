import numpy as np
import matplotlib.pyplot as plt

def rk4(System, y0, t, h):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(1, n):
        k1 = System(y[i - 1]) * h
        k2 = System(y[i - 1] + k1 / 2) * h
        k3 = System(y[i - 1] + k2 / 2) * h
        k4 = System(y[i - 1] + k3) * h
        y[i] = y[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    return y

def Lorenz_Attractor(r):
    sigma = 10
    R = 28
    beta = 8 / 3
    x, y, z = r
    Xprime = sigma * (y - x)
    Yprime = x * (R - z) - y
    Zprime = x * y - beta * z

    return np.array([Xprime, Yprime, Zprime])

t0, t_end = 0, 50
h = 1e-3
t = np.arange(t0, t_end, h)
y0 = np.array([1, 1, 1])
y1 = np.array([1 + 1e-9, 1, 1])

r = rk4(Lorenz_Attractor, y0, t, h)
r_1 = rk4(Lorenz_Attractor, y1, t, h)

# Plot the 3D trajectory
plt.figure(figsize=(10, 7))
plt.plot(t, r[:, 0], color="g", label="Initial Condition: (1,1,1)")
plt.plot(t, r_1[:, 0], color="b", label="Perturbed: (1+1e-9,1,1)")
plt.title("Lorenz Attractor - X Component Over Time")
plt.xlabel("Time t")
plt.ylabel("x(t)")
plt.legend()
plt.show()

