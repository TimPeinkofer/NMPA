
def func(x): # function need to evaluate
    return x+x**3

def array_filling(h,a,b,n,f, array):
    
    if n != len(array):
        raise ValueError("Please choose matching values for calculation") #Compare the size of the array and the Number of data points
    
    for i in range(n): # Calculate the value for every datapoint until the upper bound is reached
        x = a +i*h

        if x > b:
            break

        array[i] = f(x)

    return array


def Newton_cotes(n,x, f):

    h = (x[-1] - x[0]) / (n - 1)  # Calculating the stepsize h
    sum_integral = 0

    for i in range(0, n - 2, 2):  # Calculating the value for every odd step
        sum_1 = f[i] + 4 * f[i + 1] + f[i + 2]
        sum_integral += h/3 * sum_1

    return sum_integral


n = 5
h = 1/4
a = [0]*n
x = [0,1]

array = array_filling(h,0,1,n,func,a)

result = Newton_cotes(5,x,array)
print(f"Result of the Integration: {result}")