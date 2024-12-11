import numpy as np
import matplotlib.pyplot as plt

#Functions
def f1(x):
    return np.exp(x)*np.cos(x)
def f2(x):
    return np.exp(x)

def f31(x):
        return np.exp(2*x)
def f32(x):
        return x-2*np.cos(x)-4
#Antiderivative
def F1(x):
    return np.exp(x)*0.5*(np.cos(x)+np.sin(x))
def F2(x):
    return np.exp(x)
def F31(x):
        return 0.5*np.exp(2*x)
def F32(x):
        return -2*np.sin(x)+0.5*x**2+4*x


#Trapezoidal rule 
def Trapezoidal(a,b,n,f):
    x=np.linspace(a,b,n+1)
    m=0
    h=(b-a)/n
    y=f(x)

    for i in range(1,n):
        m += y[i]

    I=h/2*(y[0]+2*m+y[-1])
    return I

#Simpson 3/8 Rule with check if n is not multible of 3

def Simpson(a,b,n,f):

    #divide the intervall n into two, where n1 is multible of 3
    n1 = (n // 3) * 3
    n2 = n - n1
    
    x=np.linspace(a,b,n+1)
    m1,m2,m3=0,0,0
    y=f(x)

    #h for the intervall n1 
    h1 = (x[n1] - a) / n1  
    #h for the intervall n2
    if n2 > 0:
        h2 = (b - x[n1]) / n2   
    else:
        h2 =0

    #calculate m1 
    for i in range(1,n1,3):
        m1 += y[i]
    #calculate m2
    for i in range(2,n1,3):
        m2 += y[i]
    #calculate m3
    for i in range(3,n1,3):
        m3 += y[i]

    I_Simpson=(3*h1/8)*(y[0]+ 3*m1 + 3*m2 + 2*m3 +y[n1])

    # calculate the rest with trapezoidal rule 
    if n2 > 0:
        I_Trapezoid = (h2 / 2) * (y[n1] + y[-1])
    else:
         I_Trapezoid = 0

    return I_Simpson + I_Trapezoid

#calculate Error with trapezoida rule

def ErrorTrap (a,b,f,F):
    Error = []
    hs= []
    numerical_value = 0
    Error_Calculation = 0
    for i in range(5,501):
        numerical_value = Trapezoidal(a,b,i,f)
        Error_Calculation = (F(b) - F(a)) - numerical_value
        Error.append(Error_Calculation)
        h= (b-a)/i
        hs.append(h)
    return Error,hs

def ErrorSimpson (a,b,f,F):
    Error = []
    hs= []
    numerical_value = 0
    Error_Calculation = 0
    for i in range(5,501):
        numerical_value = Simpson(a,b,i,f)
        Error_Calculation = (F(b) - F(a)) - numerical_value
        Error.append(Error_Calculation)
        h= (b-a)/i
        hs.append(h)
    return Error,hs

#TrapezoidalErrors
#f1
ET1,hT1= ErrorTrap (0,np.pi/2,f1,F1)
ET1 = np.log(np.abs(ET1)+ 1e-15)
hT1=np.log(hT1)
#f2
ET2,hT2=ErrorTrap(-1,3,f2,F2)
ET2 = np.log(np.abs(ET2)+ 1e-15)
hT2=np.log(hT2)
#f3  where '31' is for the intervall [-1,0] and '32' for the intervall[0,1]
ET31,hT31=ErrorTrap(-1,0,f31,F31)
ET31 = np.log(np.abs(ET31)+ 1e-15)
hT31=np.log(hT31)

ET32,hT32=ErrorTrap(0,1,f32,F32)
ET32 = np.log(np.abs(ET32)+ 1e-15)
hT32=np.log(hT32)

#SimpsonErrors
#f1
ES1,hS1= ErrorSimpson (0,np.pi/2,f1,F1)
ES1 = np.log(np.abs(ES1)+ 1e-15)
hS1=np.log(hS1)
#f2
ES2,hS2=ErrorSimpson(-1,3,f2,F2)
ES2 = np.log(np.abs(ES2)+ 1e-15)
hS2=np.log(hS2)
#f3  where '31' is for the intervall [-1,0] and '32' for the intervall[0,1]
ES31,hS31=ErrorSimpson(-1,0,f31,F31)
ES31 = np.log(np.abs(ES31)+ 1e-15)
hS31=np.log(hS31)

ES32,hS32=ErrorSimpson(0,1,f32,F32)
ES32 = np.log(np.abs(ES32)+ 1e-15)
hS32=np.log(hS32)


plt.figure(figsize=(8, 6))
plt.scatter(hT1, ET1, marker='x',color='g', label='Error for different h', zorder=3)
plt.title("Error for function 1 with Trapezoidal rule")
plt.xlabel('$log(h)$')
plt.ylabel('log(Error)')
plt.grid()
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(hT2, ET2, marker='x',color='g', label='Error for different h', zorder=3)
plt.title("Error for function 2 with Trapezoidal rule")
plt.xlabel('$log(h)$')
plt.ylabel('log(Error)')
plt.grid()
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(hT31, ET31, marker='x',color='g', label='Error for different h on [-1,0]', zorder=3)
plt.scatter(hT32, ET32, marker='x',color='b', label='Error for different h on [0,1]', zorder=3)
plt.title("Error for function 3 with Trapezoidal rule")
plt.xlabel('$log(h)$')
plt.ylabel('log(Error)')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(hS1, ES1, marker='x',color='g', label='Error for different h', zorder=3)
plt.title("Error for function 1 with Simpson 3/8 rule")
plt.xlabel('$log(h)$')
plt.ylabel('log(Error)')
plt.grid()
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(hS2, ES2, marker='x',color='g', label='Error for different h', zorder=3)
plt.title("Error for function 2 with Simpson 3/8 rule")
plt.xlabel('$log(h)$')
plt.ylabel('log(Error)')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(hS31, ES31, marker='x',color='g', label='Error for different h on [-1,0]', zorder=3)
plt.scatter(hS32, ES32, marker='x',color='b', label='Error for different h on [0,1]', zorder=3)
plt.title("Error for function 3 with Simpson 3/8 rule")
plt.xlabel('$log(h)$')
plt.ylabel('log(Error)')
plt.grid()
plt.legend()
plt.show()