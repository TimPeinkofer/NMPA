import numpy as np

M = np.array([[6, 5, -5],
              [2, 6, -2],
              [2, 5, -1]])
V = np.array([10, 11, 12])  
size,size = M.shape
iteration = 100

def matrix_vector(N, Matrix, Vector):
    Solution = np.zeros(N)
    for i in range(N):
        for j in range(N):
            Solution[i] += Matrix[i][j] * Vector[j]
    return Solution

def Eigenvalue(mat,vec, Iteration):  
    for i in range(Iteration):
        vec = matrix_vector(size, mat, vec)
    eigenvalue = vec / np.dot(vec, vec)
    eigenvector = vec
    return eigenvalue,eigenvector


eigenvalue_1,eigenvector_1 = Eigenvalue(M,V, iteration)
eigenvalue_2,eigenvector_2 = Eigenvalue(np.linalg.inv(M),V, iteration)
eigenvalue_2 = 1/eigenvalue_2
print("Largest eigenvalue and corresponding eigenvektor:", eigenvalue_1, eigenvector_1)
print("Smallest eigenvalue and corresponding eigenvektor:", eigenvalue_2, eigenvector_2)




