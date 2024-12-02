import numpy as np
import matplotlib.pyplot as plt


# define functions
def H(x):
    return 0.5*(1 + np.tanh(5*x))

def P(rho):
    return H(5-rho) *20*rho**(4/3) + H(rho-5) * 1 * rho**(5/3)
##############################################################

#define lagrange interpolation 
def LagrangeInterpolation (x,xs,ys):
    interpolated_value = 0
    for n in range(len(xs)):
      interpolated_value = interpolated_value + ys[n] * LagrangeCoeffiecient(n, x, xs,ys)
    return interpolated_value

def LagrangeCoeffiecient(n, x, xs,ys):
    denominator = 1
    numerator = 1
    
    for i in range(len(xs)):
        if i != n:
            ys = xs[n] - xs[i]
            if ys != 0:  
                denominator = denominator * ys
                numerator = numerator * (x - xs[i])
    
    return numerator / denominator
#######################################################################

#define Chi-Error
def ChiErrorPlot(rho_0, rho_1, r, n_values):
    error_log = []
    for n in n_values:
        xs = np.linspace(rho_0, rho_1, n)
        ys = [P(x) for x in xs]
        yk = [LagrangeInterpolation(x, xs, ys) for x in r]
        error = ChiError(r, P(r), yk)
        error_log.append(np.log(error))  
    return error_log

def ChiError(r, true_values, interpolated_values):
    m = len(r)
    error = np.sqrt((1 / m) * np.sum((true_values - interpolated_values)**2))
    return error
###########################################################################

def Plot (rho_0,rho_1, n,m,):
    #known values
    xs = np.linspace(rho_0,rho_1,n)  
    ys = [P(x) for x in xs]
    #interpolated values
    xk= np.linspace(rho_0,rho_1,m)
    yk=[ LagrangeInterpolation(x,xs,ys) for x in xk]
    return xk,yk,xs,ys

rho_0 , rho_1,n,m = 0, 10,5,200       #define intervall length + numbers of known values n + number of interpolated values m

#interpolation plot values
xk,yk,xs,ys = Plot (rho_0, rho_1, n, m)   
#true function       
r =np.linspace(rho_0,rho_1,200)
Pr= P(r)

#error Plot values
values_n = range(3,41)
LogE = ChiErrorPlot (rho_0,rho_1,r,values_n)

#plot function, known and predicted values
plt.figure(figsize=(8, 6))
plt.plot(r, Pr , color='black', label='true function', zorder=2)
plt.plot(xk,yk , marker='x',  color='r', label='interpolated points', zorder=1)
plt.scatter(xs, ys, color='b', label='known points', zorder=3)
plt.xlabel('$rho$')
plt.ylabel('P($rho$)')
plt.grid()
plt.legend()
plt.show()
#Plot the error  
plt.figure(figsize=(10, 6))
plt.plot(values_n, LogE, marker="o", label="Log(Error)")
plt.xlabel("Number of known Points")
plt.ylabel("Log(Error)")
plt.legend()
plt.grid()
plt.show()