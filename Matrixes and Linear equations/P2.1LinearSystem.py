import numpy as np


M = np.array([[1,2,3],
                   [0,5,6],
                   [0,0,9]])
V = [10,11,12]



def matrix_vektor (N,Matrix,Vektor):
    Solution = np.zeros(N)
    for i in range(N):
      for j in range(N):
         Solution[i] = Solution[i]+ Matrix[i][j]*Vektor[j]
    return Solution

def upper_matrix(n,Matrix,Vektor):
    solution = np.zeros(n)
    for k in range(n-1, -1, -1):
      non_diag = 0
      for l in range(k+1,n):
        non_diag = non_diag + Matrix[k][l] * solution[l] 
      solution[k] =(Vektor[k]-non_diag)/Matrix[k][k]
    return solution

print(upper_matrix(3,M,V))
print(matrix_vektor(3,M,V))
