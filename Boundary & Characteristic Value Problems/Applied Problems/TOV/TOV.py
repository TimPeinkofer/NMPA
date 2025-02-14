import numpy as np
import matplotlib.pyplot as plt

def tov_equations(y, r, K, Gamma):
    """System der TOV-Gleichungen in geometrischen Einheiten."""
    P, m = y  
    if P <= 0:  # Außerhalb des Sterns verschwindet der Druck
        return np.array([0, 0])
    
    rho_0 = (P / K) ** (1 / Gamma)
    epsilon = P / (rho_0 * (Gamma - 1))
    rho = rho_0 * (1 + epsilon)
    
    if r < 1e-6:
        return np.array([0, 0])
    
    denom = r**2 - 2 * m * r
    if denom <= 1e-10:  #singularity save
        return np.array([0, 0])
    
    P_prime = - (rho + P) * (m + 4 * np.pi * r**3 * P) / denom
    m_prime = 4 * np.pi * r**2 * rho
    
    return np.array([P_prime, m_prime])

def rk4(system, y0, t, h, K, Gamma):
    """Runge-Kutta 4. Ordnung (RK4) für das System."""
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(1, n):
        k1 = h * system(y[i - 1], t[i - 1], K, Gamma)
        k2 = h * system(y[i - 1] + k1 / 2, t[i - 1] + h / 2, K, Gamma)
        k3 = h * system(y[i - 1] + k2 / 2, t[i - 1] + h / 2, K, Gamma)
        k4 = h * system(y[i - 1] + k3, t[i - 1] + h, K, Gamma)
        y[i] = y[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        if y[i, 0] <= 0:
            y[i:] = 0
            break
    
    return y

def make_star(rhoe, K, Gamma, r_max=10, dr=1e-3):
    """Erzeugung des Sterns in geometrischen Einheiten."""
    P_central = K * rhoe ** Gamma  # central preasure 
    r = np.arange(dr, r_max, dr)
    y0 = np.array([P_central, 0])
    y = rk4(tov_equations, y0, r, dr, K, Gamma)
    P, m = y[:, 0], y[:, 1]
    
    surface_idx = np.argmax(P <= 0) - 1  # Last Index
    R, M = r[surface_idx], m[surface_idx]   
    return r, P, m, R, M

# Parameters in geometrik units 
K = 3000 
Gamma = 2.5 
rhoe = 8e-4 

r, P, m, R, M = make_star(rhoe, K, Gamma)

print(f"Sterndurchmesser: R = {R:.4f} ")
print(f"Gravitationsmasse: M = {M:.4f} ")
plt.figure(figsize=(6, 4))
plt.plot(r, P, label='Preassure P(r)')
plt.xlabel('Radius')
plt.ylabel('Preassure')
plt.title('Preassure vs. Radius')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(r, m, label='Masse m(r)', color='red')
plt.xlabel('Radius')
plt.ylabel('Mass')
plt.title('Mass vs. Radius')
plt.legend()
plt.grid()
plt.show()

Gamma_values = np.linspace(1.1, 5, 100)
K_values = np.linspace(100, 5000, 100)
R_vs_Gamma, M_vs_Gamma = [], []
R_vs_K, M_vs_K = [], []

for gamma in Gamma_values:
    r, P, m, R, M = make_star(rhoe, K, gamma)
    R_vs_Gamma.append(R)
    M_vs_Gamma.append(M)

for k in K_values:
    r, P, m, R, M = make_star(rhoe, k, Gamma)
    R_vs_K.append(R)
    M_vs_K.append(M)

plt.figure(figsize=(6, 4))
plt.plot(Gamma_values, R_vs_Gamma, label='R vs Gamma')
plt.plot(Gamma_values, M_vs_Gamma, label='M vs Gamma')
plt.xlabel('Gamma')
plt.ylabel('R, M')
plt.title('R and M over Gamma')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(K_values, R_vs_K, label='R vs K')
plt.plot(K_values, M_vs_K, label='M vs K')
plt.xlabel('K')
plt.ylabel('R, M')
plt.title('R and M over K')
plt.legend()
plt.grid()
plt.show()

