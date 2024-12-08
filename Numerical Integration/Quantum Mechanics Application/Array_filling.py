def array_filling(h,a,b,n,f, array):
    
    if n != len(array):
        raise ValueError("Please choose matching values for calculation") #Compare the size of the array and the Number of data points
    
    for i in range(n): # Calculate the value for every datapoint until the upper bound is reached
        x = a +i*h

        if x > b:
            break

        array[i] = f(x)

    return array

#Testing case, will be deleted in the Application part
def func(x):
    return x

h = 1/100
a= [0]*100
result = array_filling(h,0,1,100,func,a)
print(f"Result array: {result}")
