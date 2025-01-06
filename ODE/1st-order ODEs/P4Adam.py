import numpy as np
import matplotlib.pyplot as plt


def Fprime(x, y):
    yprime = np.sin(x*y)*np.cos(x+y)
    #yprime = y * np.cos(x + y)
    return yprime

def Adam(x0,xm,y0,n,f):
  h=(xm-x0)/n
  x=np.arange(x0,xm+h,h)
  y=np.zeros(len(x))

  y[0] = y0
  y[1] = y[0] + h * Fprime(x[0], y[0])  #first two values via heun
  y[2] = y[1] + h * Fprime(x[1], y[1])
  for i in range(3, len(x)):     #Adam
        k = y[i-1] + (h / 12) * ( 5 * f(x[i-1], y[i-1]) + 8 * f(x[i-2], y[i-2]) - f(x[i-3], y[i-3]))
        y[i]=k
  return x,y

x0=0
xm=30
n=1000
y0=1

x,f=Adam(x0,xm,y0,n,Fprime)

plt.figure(figsize=(16, 9))
plt.plot(x, f)
plt.title("Numerical solution Adam")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()