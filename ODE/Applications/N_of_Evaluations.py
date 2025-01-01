import numpy as np
import matplotlib.pyplot as plt
from Adams_multon import adam_predictor, adams_corrector, f
from Runge_Kutta_Task_2 import runge_kutta  


N_values = np.linspace(10, 5000, 100, dtype=int)
errors_1 = []
errors_2 = []
errors_3 = []
evaluations_1 = []  # For Adams with corrector
evaluations_2 = []  # For Adams without corrector
evaluations_3 = []  # For Runge-Kutta

for i in range(len(N_values)-1):  # calculate the error for every N and 2N
    N_v = N_values[i]
    N_2 = N_values[i + 1]

   
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
    y_runge_N_2 = runge_kutta(f, 0.1, 0, 2, N_2)[1]  # Only get the solution values
    y_runge_N = runge_kutta(f, 0.1, 0, 2, N_v)[1]  # Only get the solution values

    # Compute the absolute error at x = 2 
    error_1 = abs(y_corr_N[-1] - y_corr_N_2[-1])  # Error between N and 2N for Adams with corrector
    errors_1.append(error_1)
    evaluations_1.append(2 * N_v)  # 2N evaluations for Adams with corrector
    
    error_2 = abs(y_pred_N[-1] - y_corr_N_2[-1])  # Error between N and 2N for Adams without corrector
    errors_2.append(error_2)
    evaluations_2.append(N_v)  # N evaluations for Adams without corrector
    
    error_3 = abs(y_runge_N[-1] - y_runge_N_2[-1])  # Error between N and 2N for Runge-Kutta
    errors_3.append(error_3)
    evaluations_3.append(4 * N_v)  # 4N evaluations for Runge-Kutta

# Plot the error vs function evaluations
plt.figure(figsize=(10, 6))
plt.loglog(evaluations_1, errors_1, linestyle='-', color='b', label="Adams with corrector (2N)")
plt.loglog(evaluations_2, errors_2, linestyle='-', color='r', label="Adams without corrector (N)")
plt.loglog(evaluations_3, errors_3, linestyle='-', color='g', label="Runge-Kutta (4N)")
plt.title("Error vs Function Evaluations")
plt.xlabel("Number of function evaluations")
plt.ylabel("Absolute error")
plt.legend()
plt.grid(True)
plt.show()
