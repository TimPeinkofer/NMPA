import numpy as np
import matplotlib.pyplot as plt
from Adams_multon import adam_predictor, adams_corrector, f
from Runge_Kutta_Task_2 import runge_kutta

# Define N as an array of integers (10 to 5000 steps)
N_values = np.linspace(10, 5000, 100, dtype=int)
errors_1 = []
errors_2 = []
errors_3 = []

for i in range(len(N_values)-1):  # calculate the error for every N and 2N
    N_v = N_values[i]
    N_2 = N_values[i + 1]

    # Create grids for N and 2N
    x_N = np.linspace(0, 2, N_v)
    x_N_2 = np.linspace(0, 2, N_2)
    h_N = x_N[1] - x_N[0]
    h_N_2 = x_N_2[1] - x_N_2[0]

    # Compute the values via Adams-Moulton method
    y_pred_N = adam_predictor(f, 0.1, x_N, N_v, h_N)
    y_corr_N = adams_corrector(f, y_pred_N, x_N, N_v, h_N)

    y_pred_N_2 = adam_predictor(f, 0.1, x_N_2, N_2, h_N_2)
    y_corr_N_2 = adams_corrector(f, y_pred_N_2, x_N_2, N_2, h_N_2)

    # Compute the values via Runge-Kutta method
    y_runge_N_2 = runge_kutta(f, 0.1, 0, 2, N_2)[1]
    y_runge_N = runge_kutta(f, 0.1, 0, 2, N_v)[1]

    # Compute the absolute error at x = 2 
    error_1 = abs(y_corr_N[-1] - y_corr_N_2[-1])  # Error between N and 2N for Adams with corrector
    errors_1.append(error_1)
    
    error_2 = abs(y_pred_N[-1] - y_corr_N_2[-1])  # Error between N and 2N for Adams without corrector
    errors_2.append(error_2)
    
    error_3 = abs(y_runge_N[-1] - y_runge_N_2[-1])  # Error between N and 2N for Runge-Kutta
    errors_3.append(error_3)

# Plot the absolute error in a log-log plot
plt.figure(figsize=(10, 6))
plt.loglog(N_values[:-1], errors_1, linestyle='-', color='b', label="Adams with corrector")
plt.loglog(N_values[:-1], errors_3, linestyle='-', color='g', label="Runge-Kutta")
plt.loglog(N_values[:-1], errors_2, linestyle='--', color='r', label="Adams without corrector")
plt.title("Absolute Error of Different Methods")
plt.xlabel("Number of steps (N)")
plt.ylabel("Absolute error (E)")
plt.legend()
plt.grid(True)
plt.show()
