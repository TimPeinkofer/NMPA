import numpy as np
import scipy.linalg as linalg

# Lane-Emden equation
def Lane_Emden_system(y, xi, n):
    theta, dtheta_dxi = y
    if xi == 0:
        return np.array([dtheta_dxi, 0])  # Avoid division by zero at xi=0
    return np.array([dtheta_dxi, - (2 / xi) * dtheta_dxi - theta ** n])

# Runge-Kutta 4 (RK4)
def rk4(System, y0, xi_max, h, n):
    xi_values = np.arange(1E-4, xi_max, h)  
    y = np.zeros((len(xi_values), len(y0)))
    y[0] = y0
    
    for i in range(1, len(xi_values)):
        xi = xi_values[i - 1]
        k1 = System(y[i - 1], xi, n)
        k2 = System(y[i - 1] + h * k1 / 2, xi + h / 2, n)
        k3 = System(y[i - 1] + h * k2 / 2, xi + h / 2, n)
        k4 = System(y[i - 1] + h * k3, xi + h, n)
        y[i] = y[i - 1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    return xi_values, y[:, 0], y[:, 1]  

# Initial conditions
y0 = np.array([1, 0])  
xi_max = 10
h = 1  
n = 1

xi_vals, theta_vals, dtheta_vals = rk4(Lane_Emden_system, y0, xi_max, h, n)

# Densities and pressures calculation
def get_values_dens_press(theta_vals, n, dens_c=1, K=1):
    dens_vals = dens_c * theta_vals**n
    press_vals = K * dens_vals**(1 + 1/n)
    return dens_vals, press_vals

dens_vals, press_vals = get_values_dens_press(theta_vals, n)

# Ensure density values are positive for log computation
dens_vals = np.where(dens_vals > 0, dens_vals, 1e-10)
press_vals = np.where(press_vals > 0, press_vals, 1e-10)

# Calculation of k(r)
def k_values(r, Gamma_1, P, h):
    k = np.zeros(len(r)-1)  # Initialize array with correct size
    for i in range(1, len(r)-1):  # Avoid boundaries using central difference
        k[i] = 2/r[i] + 1/(Gamma_1[i] * P[i]) * (P[i+1] - P[i-1]) / (2*h)
    return k

def gamma_values(r, P,dens_val):
    k = np.zeros(len(r)-1)  # Initialize array with correct size
    for i in range(1, len(r)-1):  # Avoid boundaries using central difference
        k[i] = (np.log(P[i+1])-np.log(P[i-1]))/(np.log(dens_val[i+1])-np.log(dens_val[i-1]))
    return k

# Calculation of matrices A and B for the oscillatory equation
def compute_AB(xi_vals, press_vals, dens_vals, h):
    N = len(xi_vals)
    # Erstelle Matrizen für die inneren Punkte: i = 1 ... N-2
    A = np.zeros((N-2, N-2))
    B = np.zeros((N-2, N-2))
    
    # Berechne k(r) an den inneren Punkten wie bisher (für i = 1, ..., N-2)
    gamma1 = gamma_values(xi_vals,press_vals,dens_vals)
    k = k_values(xi_vals, gamma1, press_vals, h)
    
    # Für alle inneren Punkte bis auf den letzten
    for i in range(1, N-2):
        r = xi_vals[i]
        # Standarddifferenzen für das Differentialgleichungssystem
        if i-2 >= 0:
            A[i-1, i-2] = 1/h**2 - k[i] / (2*h)
        A[i-1, i-1] = -2/h**2 - 4/r**2
        if i < N-2-1:
            A[i-1, i] = 1/h**2 + k[i] / (2*h)
        B[i-1, i-1] = dens_vals[i] / (gamma1[i] * press_vals[i])
    
   
    i = N - 2  
    # Berechne rho0 am Rand:
    rho_surf = dens_vals[-1]
    rho0_surf = (dens_vals[-1] - dens_vals[-2]) / h
   
    A[-1, -2] = -1/h
    A[-1, -1] = 1/h + rho0_surf/rho_surf
    B[-1, -1] = 0
    
    return A, B


A, B = compute_AB(xi_vals, press_vals, dens_vals, h)

try:
    w,v = linalg.eig(A, B)
except linalg.LinAlgError:
    print("Error in eigenvalue computation")

print(w)
print(v)
