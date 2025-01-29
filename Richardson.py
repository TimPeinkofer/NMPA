import numpy as np

def func(x):
    return np.arctan(x)

# Approximation of the function values
def F(h, x):
    return 1/(2*h)*(func(x + h) - func(x - h)) 

# Define function for Richardson extrapolation
def psi(h, x):
    return (4 * F(h / 2, x) - F(h, x)) / 3

# Richardson Extrapolation
def Richardson(x, h):
    value = psi(h, x)
    error = abs(value - psi(h / 2, x))  # Estimate error of our calculation
    return value, error

# Calculate derivative via Richradson extrapolation
result, error = Richardson(0, np.pi / 6)
print(f"Richardson Extrapolated Value: {result}")
print(f"Estimated Error: {error}")
