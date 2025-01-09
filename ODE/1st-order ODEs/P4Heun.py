import numpy as np
import matplotlib.pyplot as plt

#calcluta de two values of the given function
def Fprime(x, y):
    yprime = np.sin(x*y)*np.cos(x+y)
    #yprime = y * np.cos(x + y)
    return yprime

def Heun(x0,xm,y0,n,f):
  h=(xm-x0)/n
  x=np.arange(x0,xm+h,h)
  y=np.zeros(len(x))

  y[0] = y0
  for i in range(1,len(x)):
        ypred = y[i-1] + h * f(x[i-1], y[i-1])  
        ycorr = y[i-1] + h/2 * (f(x[i-1], y[i-1]) + f(x[i],ypred))
        y[i]=ycorr 
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
