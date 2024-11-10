import numpy as np

# Generate a random NxN matrix and vector
n = 2  
A = np.round(np.random.rand(n, n), 2)  
b = np.round(np.random.rand(n, 1), 2)


print("Matrix:")
print(A)
print("Vector:")
print(b)
print("\n")

Max_iterations = 15000

# Get dimensions of our NxN matrix
rows, columns = A.shape


def overrel(m, vector, iterations, w, tol=1e-3):  # Overrelaxation function
    x = vector.copy()  # Use a copy of the initial guess to avoid modifying the original
    for j in range(iterations):
        x_old = x.copy()
        for i in range(rows):
            if m[i][i] != 0:
                # Calculate the factor for iteration
                factor = w / m[i][i]
                
                # Calculate the sums for the iteration part
                sum1 = sum(m[i][l] * x[l] for l in range(i))
                sum2 = sum(m[i][l] * x_old[l] for l in range(i, rows))
                
                # Update x based on our sums and the factor
                x[i] = x_old[i] + factor * (vector[i] - sum1 - sum2)

        # Check for convergence
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            print(f"Convergence achieved after {j + 1} iterations with w = {w}")
            return x

    print(f"No convergence after {iterations} iterations with w = {w}")
    return None

# Calculate the function with different overrelaxation factors
for factor in np.arange(1, 2, 0.1):  
    result = overrel(A, b, Max_iterations, factor)
    if result is not None:
        print(f"Solution vector after over-relaxation (w = {factor}):")
        print(result)
        print(" ")
