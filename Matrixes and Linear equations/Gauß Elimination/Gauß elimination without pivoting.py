import numpy as np

# Initialize matrix and vector via numpy
matrix = np.array([[1, 2, 3], [4, 5, 7], [7, 8, 9]], dtype=np.float64)
vector = np.array([10, 11, 12], dtype=np.float64).reshape(-1, 1)

print("Original Matrix:") # print the original matrix as a reference
print(matrix)
print("Original Vector:")
print(vector)

# Get dimensions of our matrix for later use
rows, columns = matrix.shape


def gauss(): # Function for gauß elimination
    
    # Generate a copy of the vector and the matrix for our gauß algorithm
    U_Matrix = np.copy(matrix)
    U_vector = np.copy(vector)
    x = np.zeros((rows, 1)) # Generate a solution vektor based on the number of rows of our matrix
    
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
                factor = U_Matrix[j][i] / U_Matrix[i][i]
                U_Matrix[j] = U_Matrix[j] - factor * U_Matrix[i]
                U_vector[j] = U_vector[j] - factor * U_vector[i]
    return U_Matrix, U_vector

def solver(mat, vec): # Solver from our first program (a little bit modified)
    for i in range(rows):
        index = rows - i - 1
        b_new = vec[index] / mat[index, index]
        
        for r in range(index + 1, rows):
            b_new -= mat[index, r] * x[r] / mat[index, index]
        
        x[index] = b_new
    return x

# Get the triangular matrix and solve the linear equation
m, v = gauss()
solution = solver(m, v)

print("Upper Triangular Matrix:")
print(m)
print("Modified Vector after Gaussian elimination:")
print(v)
print("Solution Vector:")
print(solution)
