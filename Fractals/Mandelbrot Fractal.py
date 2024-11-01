import numpy as np
import matplotlib.pyplot as plt

def find_roots_in_grid(N=400, max_steps=200, x_range=(-2, 0.7), y_range=(-1.5, 1.2)):
    x_values = np.linspace(x_range[0], x_range[1], N)
    y_values = np.linspace(y_range[0], y_range[1], N)
    
    mandelbrot_set = np.zeros((N, N), dtype=complex)  # Speicherung der komplexen Werte
    iterations_grid = np.zeros((N, N))  # Speicherung der Anzahl der Iterationen

    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            c = complex(x, y)
            z = 0
            iter_count = 0

            # Mandelbrot-Iteration
            while abs(z) < 2 and iter_count < max_steps:
                z = z**2 + c
                iter_count += 1
            
            # Speichere die Anzahl der Iterationen und den Endwert
            mandelbrot_set[i, j] = z
            iterations_grid[i, j] = iter_count

    return mandelbrot_set, iterations_grid


def plot_imaginary_part(grid_roots, x_range=(-2, 0.7), y_range=(-1.5, 1.2)):
    plt.figure(figsize=(8, 8))
    plt.imshow(np.imag(grid_roots), extent=(x_range[0], x_range[1], y_range[0], y_range[1]), cmap="inferno")
    plt.colorbar(label="Im(z)")
    plt.ylabel("Im(z)")
    plt.xlabel("Re(z)")
    plt.title("Imaginary part of roots in (x, y) plane")
    plt.show()

def plot_log_iterations(iterations_grid, x_range=(-2, 0.7), y_range=(-1.5, 1.2)):
    log_iterations = np.log10(iterations_grid, where=iterations_grid > 0)  # Logarithmus Basis 10
    plt.figure(figsize=(8, 8))
    plt.imshow(log_iterations, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), cmap="viridis")
    plt.colorbar(label="log10(Number of Iterations)")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title("Logarithm of iterations in (x, y) plane")
    plt.show()


if __name__ == "__main__":
    grid_roots, iterations_grid = find_roots_in_grid()
    plot_imaginary_part(grid_roots)
    plot_log_iterations(iterations_grid)
