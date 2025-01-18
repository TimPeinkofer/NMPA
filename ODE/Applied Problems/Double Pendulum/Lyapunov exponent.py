import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
g = 9.8  
L1 = 1  
L2 = 1  
m1 = 1  
m2 = 1  

# Function to calculate Lyapunov exponent
def lyapunov_exponent(t, sol, epsilon_0):
    # Calculate the logarithmic growth of the separation
    return np.log(sol / epsilon_0) / t

def double_pendulum(t, state): 
    theta1, theta2, omega1, omega2 = state
    
    # Definition of the ODEs
    dtheta1_dt = omega1
    dtheta2_dt = omega2

    denom1 = L1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2))
    denom2 = L2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2))
    
    omega1_dot = (-g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2)
                  - 2 * np.sin(theta1 - theta2) * m2 * (omega2 ** 2 * L2 + omega1 ** 2 * L1 * np.cos(theta1 - theta2))
                  ) / denom1
    
    omega2_dot = (2 * np.sin(theta1 - theta2) * (omega1 ** 2 * L1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1)
                  + omega2 ** 2 * L2 * m2 * np.cos(theta1 - theta2))) / denom2

    return np.array([dtheta1_dt, dtheta2_dt, omega1_dot, omega2_dot]) 

# 4th Order Runge Kutta
def runge_kutta(f, y_0, t_start, t_end, n):
    h = (t_end - t_start) / n  # Stepsize
    t_values = np.linspace(t_start, t_end, n + 1)
    y_values = np.zeros((n + 1, len(y_0)))
    y_values[0] = y_0

    for i in range(n):
        t_i = t_values[i]
        y_i = y_values[i]

        k1 = h * f(t_i, y_i)
        k2 = h * f(t_i + 0.5 * h, y_i + 0.5 * k1)
        k3 = h * f(t_i + 0.5 * h, y_i + 0.5 * k2)
        k4 = h * f(t_i + h, y_i + k3)

        y_values[i + 1] = y_i + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t_values, y_values

# Boundary conditions
theta1_0 = np.pi / 2
theta2_0 = np.pi / 2
omega1_0 = 0
omega2_0 = 0
y_0 = [theta1_0, theta2_0, omega1_0, omega2_0]

t_start = 0
t_end = 40
n = 1800
dt = (t_end - t_start) / n

# Solve for the first trajectory
t_1, result_1 = runge_kutta(double_pendulum, y_0, t_start, t_end, n)

# Perturbed initial conditions
theta1_0 = np.pi / 2 + 1e-3
theta2_0 = np.pi / 2
omega1_0 = 0
omega2_0 = 0
y_0 = [theta1_0, theta2_0, omega1_0, omega2_0]

# Solve for the second trajectory
t_2 , result_2 = runge_kutta(double_pendulum, y_0, t_start, t_end, n)

# Positions for both trajectories
theta1_1 = result_1[:, 0]
theta2_1 = result_1[:, 1]
x1_1 = L1 * np.sin(theta1_1)
y1_1 = -L1 * np.cos(theta1_1)
x2_1 = x1_1 + L2 * np.sin(theta2_1)
y2_1 = y1_1 - L2 * np.cos(theta2_1)

theta1_2 = result_2[:, 0]
theta2_2 = result_2[:, 1]
x1_2 = L1 * np.sin(theta1_2)
y1_2 = -L1 * np.cos(theta1_2)
x2_2 = x1_2 + L2 * np.sin(theta2_2)
y2_2 = y1_2 - L2 * np.cos(theta2_2)

# Separation between the two trajectories
sol = np.sqrt((x2_1 - x2_2)**2 + (y2_1 - y2_2)**2 + (x1_1 - x1_2)**2 + (y1_1 - y1_2)**2)

# Initial separation (epsilon_0)
epsilon_0 = np.sqrt((x2_1[0] - x2_2[0])**2 + (y2_1[0] - y2_2[0])**2 + (x1_1[0] - x1_2[0])**2 + (y1_1[0] - y1_2[0])**2)

# Calculate Lyapunov exponent over time
lyapunov_vals = lyapunov_exponent(t_1, sol, epsilon_0)

# Plot the Lyapunov exponent over time
print('Lyapunov Exponent:', lyapunov_vals[-1])
plt.plot(t_1, lyapunov_vals[-1]*np.ones(len(t_1)), 'r--')
plt.plot(t_1, lyapunov_vals)
plt.xlabel('Time (s)')
plt.ylabel('Lyapunov Exponent')
plt.title('Lyapunov Exponent for the Double Pendulum')
plt.show()
