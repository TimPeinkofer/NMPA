import numpy as np

M = np.array([[6, 5, -5],
              [2, 6, -2],
              [2, 5, -1]])
V = np.array([10, 11, 12])  
size,size = M.shape
iteration = 10

def Aitken(Eigenvalues):
    if len(Eigenvalues) < 3:
        return Eigenvalues[-1]  
    else:
        x_n, x_n1, x_n2 = Eigenvalues[-3], Eigenvalues[-2], Eigenvalues[-1]
        return x_n - (x_n1 - x_n) ** 2 / (x_n2 - 2 * x_n1 + x_n)

def matrix_vector(N, Matrix, Vector):
    Solution = np.zeros(N)
    for i in range(N):
        for j in range(N):
            Solution[i] += Matrix[i][j] * Vector[j]
    return Solution

def Eigenvalue(mat,vec, Iteration):  
    EVArray=[]
    for i in range(Iteration):
        vec = matrix_vector(size, mat, vec)
        eigenvector = vec / np.linalg.norm(vec)
        eigenvalue = np.dot(matrix_vector(size, mat, vec),vec) / np.dot(vec, vec)
        EVArray.append (eigenvalue)
        Aitken_Eigenvalue = Aitken(EVArray)
        if i >= 3 and  Aitken_Eigenvalue-EVArray[-1]< 1e-8  :      
            eigenvalue = Aitken_Eigenvalue
            break
        
    return eigenvalue,eigenvector


eigenvalue_1,eigenvector_1 = Eigenvalue(M,V, iteration)
eigenvalue_2,eigenvector_2 = Eigenvalue(np.linalg.inv(M),V, iteration)
eigenvalue_2 = 1/eigenvalue_2
print("Largest eigenvalue and corresponding eigenvektor:", eigenvalue_1, eigenvector_1)
print("Smallest eigenvalue and corresponding eigenvektor:", eigenvalue_2, eigenvector_2)




