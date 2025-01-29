import numpy as np

def func(x):
    return np.arctan(x)  # Funktion zur Auswertung

x = np.array([-6,-5,-4,-3,-2, -1, 0, 1, 2,3,4,5,6])  # Array mit Stützstellen
values = func(x)  # Funktionswerte
n = len(x) - 1  # Anzahl der Intervalle für die Interpolation

def h(i):
    return x[i + 1] - x[i]  # Schrittweite zwischen den Punkten

def cubic_splines():
    if n < 2:
        print("Nicht genügend Punkte für Splines")
        return None, None, None
    
    matrix = np.zeros((n - 1, n - 1), dtype=float)
    vec = np.zeros(n - 1, dtype=float)

    matrix[0, 0] = (h(0) + h(1)) * (h(0) + 2 * h(1)) / h(1)
    if n > 2:
        matrix[0, 1] = (h(1)**2 - h(0)**2) / h(1)
    
    vec[0] = (values[2] - values[1]) / h(1) - (values[1] - values[0]) / h(0)

    for i in range(1, n - 2):
        matrix[i, i - 1] = h(i)
        matrix[i, i] = 2 * (h(i) + h(i + 1))
        matrix[i, i + 1] = h(i + 1)
        vec[i] = (values[i + 2] - values[i + 1]) / h(i + 1) - (values[i + 1] - values[i]) / h(i)

    if n > 2:
        matrix[n - 2, n - 3] = (h(n - 3)**2 - h(n - 2)**2) / h(n - 3)
        matrix[n - 2, n - 2] = (h(n - 2) + h(n - 1)) * (h(n - 2) + 2 * h(n - 1)) / h(n - 1)
    
    vec[n - 2] = (values[n] - values[n - 1]) / h(n - 1) - (values[n - 1] - values[n - 2]) / h(n - 2)
    vec *= 6

    if np.linalg.cond(matrix) > 1e12:
        print("Matrix ist schlecht konditioniert, Lösung könnte ungenau sein.")
        return matrix, vec, None
    
    try:
        solution = np.linalg.solve(matrix, vec)
        S_0 = ((h(0) + h(1)) * solution[0] - h(0) * solution[1]) / h(1)
        S_n = ((h(n - 2) + h(n - 1)) * solution[n - 2] - h(n - 1) * solution[n - 3]) / h(n - 2)
        
        solution = np.append(solution, S_n)
        solution = np.insert(solution, 0, S_0)
        return matrix, vec, solution
    except np.linalg.LinAlgError:
        print("Keine Lösung vorhanden")
        return matrix, vec, None

m, v, sol = cubic_splines()

if sol is not None:
    def spline_derivative(i, x_val):
        if x[i] <= x_val <= x[i + 1]:
            a = (sol[i + 1] - sol[i]) / (6 * h(i))
            b = sol[i] / 2
            c = (values[i + 1] - values[i]) / h(i) - h(i) / 6 * (2 * sol[i] + sol[i + 1])
            return 3 * a * (x_val - x[i])**2 + 2 * b * (x_val - x[i]) + c
        else:
            print(f"x_val = {x_val} liegt außerhalb des Intervalls [{x[i]}, {x[i+1]}]")
            return None
    
    derivative_at_0 = spline_derivative(5, 0)  # Index angepasst für x=0
    if derivative_at_0 is not None:
        print(f"Approx. Ableitung von arctan(x) bei x=0: {derivative_at_0}")
        print(f"Exakte Ableitung: {1 / (1 + 0**2)}")
        print(f"Error: {1-derivative_at_0}")
