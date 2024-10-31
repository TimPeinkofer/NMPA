import numpy as np

# Definition of the functions
def func(x, y):
    return x * y - 0.1

def g_y_1(x):
    return np.sqrt(1/3*(2 - x**2))

def g_x_2(y):
    return np.sqrt(2 - 3*y**2)

def g_x_1(y):
    return 0.1/y

def g_y_2(x):
    return 0.1 / x

def negative_g_x_2(y):
    return -g_x_2(y)

def negative_g_y_1(x):
    return -g_y_1(x)

def solve_fixed_point(f1, f2, x_init, y_init, tol=1e-6, max_iter=100):
    # Initial guesses for x and y
    x, y = x_init, y_init
    for i in range(max_iter):
        # Update x and y using the fixed-point iterations
        x_new = f1(y)
        y_new = f2(x)
        
        # Check for convergence
        if abs(x_new - x) < tol and abs(y_new - y) < tol:
            print(f"Converged in {i+1} iterations.")
            return x_new, y_new
        
        x, y = x_new, y_new

    print("Did not converge.")
    return None, None

# Run the function with different initial guesses
initial_guesses = [(1.5, 0.1), (0.1, 1), (2,1)]
solutions = []
for x_init, y_init in initial_guesses:
    x_sol, y_sol = solve_fixed_point(g_x_1, g_y_1,x_init, y_init)
    if x_sol is not None and y_sol is not None:
        solutions.append((x_sol, y_sol,func(x_sol,y_sol), x_init,y_init))

for x_init, y_init in initial_guesses:
    x_sol, y_sol = solve_fixed_point(g_x_2, g_y_2,x_init, y_init)
    if x_sol is not None and y_sol is not None:
        solutions.append((x_sol, y_sol, func(x_sol,y_sol),x_init,y_init))


for x_init, y_init in initial_guesses:
    x_sol, y_sol = solve_fixed_point(g_x_1,negative_g_y_1,x_init, y_init)
    if x_sol is not None and y_sol is not None:
        solutions.append((x_sol, y_sol,func(x_sol,y_sol),x_init,y_init))

for x_init, y_init in initial_guesses:
    x_sol, y_sol = solve_fixed_point(negative_g_x_2, g_y_2,x_init, y_init)
    if x_sol is not None and y_sol is not None:
        solutions.append((x_sol, y_sol, func(x_sol,y_sol),x_init,y_init))

# Display solutions
for i, (x_sol, y_sol, per,x0,y0) in enumerate(solutions, 1):
    print(f"Solution {i}: x ≈ {x_sol}, y ≈ {y_sol}, Performance = {per}, x0 = {x0}, y0 = {y0}")
