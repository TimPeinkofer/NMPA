import numpy as np
import matplotlib.pyplot as plt

# Define the function
def func(x):
    return 35*x**9-180*x**7+378*x**5-420*x**3+315*x

# Define the derivative 
def derivative(x):
    return 315*x**8-1260*x**6+1890*x**4-1260*x**2+315

# finding the roots viay newton-raphson
def newton_raphson(x_0, f, df, tolerance=1e-9, max_iterations=200):
    x_t = x_0
    iteration = 0
    
    while abs(f(x_t)) >= tolerance and iteration < max_iterations: #Checking for convergence
        if abs(df(x_t))< 1e-12: #Solving the problem dividing by zero
            break
        x_t = x_t - f(x_t) / df(x_t)
        iteration += 1
    
    #
    return x_t if abs(f(x_t)) < tolerance else 0, iteration # This line returns the value after a convergence check



def find_roots_in_grid(f, df, N=400, epsilon=1e-9, max_steps=200, x_range=(-2, 2), y_range=(-2, 2)): # Generating grid to plot
    x_values = np.linspace(x_range[0], x_range[1], N)
    y_values = np.linspace(y_range[0], y_range[1], N)
    
    grid_roots = np.zeros((N, N), dtype=complex)# Generating a grid for iterations and convergence
    iterations_grid = np.zeros((N, N))

    for i, y in enumerate(y_values): #Generating the values for every pixel of the grid
        for j, x in enumerate(x_values):
            z = complex(x, y)
            root, iterations = newton_raphson(z, f, df, epsilon, max_steps)
            grid_roots[i, j] = root
            iterations_grid[i, j] = iterations

    return grid_roots, iterations_grid


def plot_imaginary_part(grid_roots, x_range=(-2, 2), y_range=(-2, 2)):
    plt.figure(figsize=(8, 8))
    plt.imshow(np.imag(grid_roots), extent=(x_range[0], x_range[1], y_range[0], y_range[1]), cmap="plasma")
    plt.colorbar(label="Im(k(z))")
    plt.ylabel("Re(z)")
    plt.xlabel("Im(z)")
    plt.title("Imaginary part of roots in (x, y) plane")
    plt.show()

def plot_log_iterations(iterations_grid, x_range=(-2, 2), y_range=(-2, 2)):
    log_iterations = np.log(iterations_grid, where=iterations_grid > 0) # Just plotting the values where number of iterations != 0
    plt.figure(figsize=(8, 8))
    plt.imshow(log_iterations, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), cmap="viridis")
    plt.colorbar(label="log10(Number of Iterations)")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title("Logarithm of iterations in (x, y) plane")
    plt.show()


if __name__ == "__main__":
    grid_roots, iterations_grid = find_roots_in_grid(func, derivative)
    plot_imaginary_part(grid_roots)
    plot_log_iterations(iterations_grid)
