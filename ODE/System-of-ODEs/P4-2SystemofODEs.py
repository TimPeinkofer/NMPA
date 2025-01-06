import numpy as np
import matplotlib.pyplot as plt

#define the two functions
def Fy(x, y, z):
    yprime = np.sin(y) + np.cos(z * x)
    return yprime

def Fz(x, y, z):
    if x!=0:
      zprime = np.exp(-y * x) + np.sin(z * x) / x
      return zprime
    else:
      return 0
    

####################################################################
def runge_kutta(x0, y0, z0, dx, x_end):
    #create an array of x,y and z values where the x values are known and z and y each have as many zeros as x elements
    x = np.arange(x0, x_end+dx, dx)
    y = np.zeros(len(x))
    z = np.zeros(len(x))

    #initial conditions
    y[0] = y0
    z[0] = z0

    # calculate for all x values y anb z 
    for i in range(1, len(x)):
        #k1 and l1
        k1 = Fy(x[i-1], y[i-1], z[i-1])
        l1 = Fz(x[i-1], y[i-1], z[i-1])
        k1 *= dx
        l1 *= dx

        #k2 and l2
        k2 = Fy(x[i-1] + 0.5 * dx, y[i-1] + 0.5 * k1, z[i-1] + 0.5 * l1)
        l2 = Fz(x[i-1] + 0.5 * dx, y[i-1] + 0.5 * k1, z[i-1] + 0.5 * l1)
        k2 *= dx
        l2 *= dx

        #k3 and l3
        k3 = Fy(x[i-1] + 0.5 * dx, y[i-1] + 0.5 * k2, z[i-1] + 0.5 * l2)
        l3 = Fz(x[i-1] + 0.5 * dx, y[i-1] + 0.5 * k2, z[i-1] + 0.5 * l2)
        k3 *= dx
        l3 *= dx

        #k4 and l4
        k4 = Fy(x[i-1] + dx, y[i-1] + k3, z[i-1] + l3)
        l4 = Fz(x[i-1] + dx, y[i-1] + k3, z[i-1] + l3)
        k4 *= dx
        l4 *= dx

        # Update y and z usingRunge-Kutta
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        z[i] = z[i-1] + (l1 + 2*l2 + 2*l3 + l4) / 6

    return x, y, z
######################################################


#initial conditions
x0 = -1  
y0 = 2.37 
z0 = -3.48  
#step size
dx = 0.1 
#end value
x_end = 4  

#calculate x,y,z
x, y, z = runge_kutta(x0, y0, z0, dx, x_end)

# Print table of values
for i in range(len(x)):
    print(f"x = {x[i]:.2f}, y(x) = {y[i]:.6f}, z(x) = {z[i]:.6f}")

# Plot y,z
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='y(x)')
plt.plot(x, z, label='z(x)')
plt.title('y and z vs x')
plt.xlabel('x')
plt.ylabel('Values y,z')
plt.legend()
plt.grid()
plt.show()

# Plot y vs z
plt.figure(figsize=(6, 6))
plt.plot(y, z, label='y vs z')  
plt.title('Parametric Plot: y vs z')
plt.xlabel('y')
plt.ylabel('z')
plt.legend()
plt.grid()
plt.show()


