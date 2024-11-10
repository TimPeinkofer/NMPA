import numpy as np

# Initialize matrix and vector via numpy
matrix = np.array([[1, 1, 0], [2 ,2, -2], [0, 3, 15]], dtype=np.float64)
vector = np.array([1, -2, 33], dtype=np.float64).reshape(-1, 1)

print("Original Matrix:")  # print the original matrix as a reference
print(matrix)
print("Original Vector:")
print(vector)

# Get dimensions of our matrix for later use
rows, columns = matrix.shape

# Generate a copy of the vector and the matrix for our Gaussian elimination algorithm
U_Matrix = np.copy(matrix)
U_vector = np.copy(vector)
x = np.zeros((rows, 1))  # Generate a solution vector based on the number of rows of our matrix

def gauss():
    for i in range(rows - 1):
        if U_Matrix[i][i] == 0:
            for k in range(i + 1, rows):
                if U_Matrix[k][i] != 0:
                    # Swap the rows in both U_Matrix and U_vector if the a_ii component is zero
                    U_Matrix[[i, k]] = U_Matrix[[k, i]]
                    U_vector[[i, k]] = U_vector[[k, i]]
                    break

        # Continue with elimination if a_ii != 0
        for j in range(i + 1, rows):
            if U_Matrix[i][i] != 0:
                factor = U_Matrix[j][i] / U_Matrix[i][i]  # Calculate the factor
                U_Matrix[j] -= factor * U_Matrix[i]
                U_vector[j] -= factor * U_vector[i]

    return U_Matrix, U_vector

def pivoting(m, v):
    U_Matrix = m
    U_vector = v
    for i in range(rows - 1, -1, -1):  
        if U_Matrix[i][i] != 0:
            # Normalize the pivots if a_ii != 0
            factor = U_Matrix[i][i]
            U_Matrix[i] /= factor #both vector and matrix!
            U_vector[i] /= factor
        
        # Get the matrix with only pivots
        for j in range(i - 1, -1, -1):
            if U_Matrix[j][i] != 0:
                factor = U_Matrix[j][i]
                U_Matrix[j] -= factor * U_Matrix[i]
                U_vector[j] -= factor * U_vector[i]
    
    return U_Matrix, U_vector


# Get the triangular matrix and solve the linear equation
m, v = gauss()
m, v = pivoting(m, v)
solution = v

print("Modified Matrix generated via Pivoting:")
print(m)
print("Solution Vector:")
print(solution)
