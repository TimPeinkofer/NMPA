import numpy as np

# Define the system of equations
def f1(x, y):
    return x*y - 0.1

def f2(x, y):
    return x**2 + 3*y**2 - 2

# Define the Jacobian matrix
def jacobian(x, y):
    return np.array([[y, x],
                     [2*x, 6*y]])

# Define the Newton-Raphson method
def newton_method(x0, y0, tol=1e-6, max_iter=100):
    x, y = x0, y0
    for _ in range(max_iter):
        # Evaluate the function values
        F = np.array([f1(x, y), f2(x, y)])
        
        # Check if we are close enough to the solution (Norm in R^2)
        if np.linalg.norm(F, ord=2) < tol:
            return x, y
        
        # Evaluate the Jacobian and solve to get the delta x
        J = jacobian(x, y)
        delta = np.linalg.solve(J, -F)
        
        # Update the values of x and y
        x, y = x + delta[0], y + delta[1] #update the values
    
    return x, y  # Return the solution after max iterations

# Find the solutions (using different initial guesses)
initial_guesses = [(0, -1), (-1, 0), (0, 1), (1, 0)]

solutions = []

for guess in initial_guesses:
    sol = newton_method(guess[0], guess[1])
    solutions.append(sol)

# Display the solutions
for i, solution in enumerate(solutions):
    print(f"Solution {i+1}: x = {solution[0]}, y = {solution[1]}")
