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
    for _ in range(max_iter): #ends the calculating if the function values are diverging
        # Evaluate the function values
        try:
            F = np.array([f1(x, y), f2(x, y)])
        except ZeroDivisionError:
            return None, None  # Exit if division by zero

        # Check for NaN in function evaluations
        if np.isnan(F).any():
            return None, None
        
        # Check if we are close enough to the solution
        if np.linalg.norm(F, ord=2) < tol:
            return x, y
        
        # Evaluate the Jacobian and solve for the delta
        J = jacobian(x, y)
        try: # check if the matrix is invertible
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return None
        
        # Update the values of x and y
        x, y = x + delta[0], y + delta[1]
    
    return x, y  # Return the solution after max iterations

# Find the solutions (using different initial guesses)
initial_guesses = [(0, -1), (-1, 0), (0, 1), (1, 0)]
solutions = []

for guess in initial_guesses:
    sol = newton_method(guess[0], guess[1])
    if sol[0] is not None and sol[1] is not None:
        solutions.append(sol)

# Display the solutions
for i, solution in enumerate(solutions):
    if solution is None or solution[0] is None or solution[1] is None:
        print(f"Solution {i+1}: No valid solution found.")
    else:
        print(f"Solution {i+1}: x = {solution[0]}, y = {solution[1]}")
