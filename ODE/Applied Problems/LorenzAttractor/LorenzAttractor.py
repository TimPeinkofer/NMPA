import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rk4(System, y0, t, h):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(1, n):
        k1 = System( y[i - 1])
        k2 = System( y[i - 1] + h * k1 / 2)
        k3 = System( y[i - 1] + h * k2 / 2)
        k4 = System( y[i - 1] + h * k3)
        y[i] = y[i - 1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


def Lorenz_Attractor (r):
    sigma = 10
    R = 28
    beta = 8 / 3
    x, y, z = r
    Xprime = sigma * (y - x)
    yprime = x * (R - z) - y
    Zprime = x * y - beta * z

    return np.array([Xprime, yprime, Zprime])

def Distance(y0,y1,t,h):
    r0 = rk4(Lorenz_Attractor, y0, t, h)
    r1 = rk4(Lorenz_Attractor, y1, t, h)
    distance = np.zeros(len(r0))

    for i in range(len(r0)):
        deltax = r0[i][0] - r1[i][0]  
        deltay = r0[i][1] - r1[i][1]  
        deltaz = r0[i][2] - r1[i][2]  
        distance[i] = np.sqrt(deltax**2 + deltay**2 + deltaz**2)  
    return distance


h = 1e-3
t=np.arange(0,20,h)
t50 = np.arange(0,50,h)
#First Question
y0 = np.array([5, 5, 5])
r0=rk4(Lorenz_Attractor,y0,t,h)

#Second Question Chaos
#a
y1=np.array([5, 5, 5+1e-5])
distance =Distance(y0,y1,t,h)
#b
h1 = 1e-6
h2 = 5e-4
r_h1 = rk4(Lorenz_Attractor, y0, t, h1)
r_h2 = rk4(Lorenz_Attractor, y0, t, h2)
deltax = r_h1[-1][0] - r_h2[-1][0]  
deltay = r_h1[-1][1] - r_h2[-1][1]  
deltaz = r_h1[-1][2] - r_h2[-1][2]  
distance_difference =  np.sqrt(deltax**2 + deltay**2 + deltaz**2) 

#c
y0_1 = np.array([5.0, 5.0, 5.0])
y0_2 = np.array([5.0, 5.0, 5.0 + 5e-15])
r0_c = rk4(Lorenz_Attractor, y0_1, t50, h)
r1_c = rk4(Lorenz_Attractor, y0_2, t50, h)
distance_low_res = np.linalg.norm(r0_c[-1] - r1_c[-1])



# Plot the 3D trajectory
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(r0[:, 0], r0[:, 1], r0[:, 2], lw=0.5)
ax.set_title("Lorenz Attractor")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

#plota)
plt.figure(figsize=(8, 5))
plt.plot(np.log(r0[:, 0]), np.log(distance))
plt.xlabel("log(x)")
plt.ylabel("log(d)")
plt.title("")
plt.show()

#plotb)

print(f"Difference in final distance between h=1e-6 and h=5e-4 at t=20: {distance_difference}")

#plotc)
print(f"Distance between trajectories at t = 50 with h = 5e-4: {distance_low_res}")
