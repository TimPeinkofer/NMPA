import numpy as np

# Initialize matrix and vector
matrix = np.array([[1, 2, 3], [4, 5, 7], [7, 8, 9]], dtype=np.float64)
vector = np.array([10, 11, 12], dtype=np.float64).reshape(-1, 1)

print("Original Matrix:")
print(matrix)
print("Original Vector:")
print(vector)

# Get dimensions
rows, columns = matrix.shape

# Initialize matrices
U_Matrix = np.copy(matrix)
U_vector = np.copy(vector)
x = np.zeros((rows, 1))

def gauss():
    for i in range(rows - 1):
        # Swap rows if the pivot is zero
        if U_Matrix[i][i] == 0:
            for k in range(i + 1, rows):
                if U_Matrix[k][i] != 0:
                    # Swap the rows in both U_Matrix and U_vector
                    U_Matrix[[i, k]] = U_Matrix[[k, i]]
                    U_vector[[i, k]] = U_vector[[k, i]]
                    break
        
        # Continue with elimination
        for j in range(i + 1, rows):
            if U_Matrix[i][i] != 0:
                factor = U_Matrix[j][i] / U_Matrix[i][i]
                U_Matrix[j] = U_Matrix[j] - factor * U_Matrix[i]
                U_vector[j] = U_vector[j] - factor * U_vector[i]
    return U_Matrix, U_vector

def solver(mat, vec):
    for i in range(rows):
        index = rows - i - 1
        b_new = vec[index] / mat[index, index]
        
        for r in range(index + 1, rows):
            b_new -= mat[index, r] * x[r] / mat[index, index]
        
        x[index] = b_new
    return x

# Perform row reduction to obtain upper triangular matrix
m, v = gauss()
solution = solver(m, v)

print("Upper Triangular Matrix:")
print(m)
print("Modified Vector after Gaussian elimination:")
print(v)
print("Solution Vector:")
print(solution)
