import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
g = 9.8  
L1 = 1  
L2 = 1  
m1 = 1  
m2 = 1  

def double_pendulum(t, state): # Definition of the function
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


t, result = runge_kutta(double_pendulum, y_0, t_start, t_end, n)

# Positions
theta1 = result[:, 0]
theta2 = result[:, 1]
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

# Plot
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, theta1, label="Theta1 (Angle 1)")
plt.plot(t, theta2, label="Theta2 (Angle 2)")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, result[:, 2], label="Omega1 (Velocity 1)")
plt.plot(t, result[:, 3], label="Omega2 (Velocity 2)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.legend()

plt.tight_layout()
plt.show()

# Plot und Animation
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-2 * L1 - 0.1, 2 * L1 + 0.1)
ax.set_ylim(-2 * L1 - 0.1, 2 * L1 + 0.1)

line, = ax.plot([], [], 'o-', lw=2, label="Pendelum")  
path, = ax.plot([], [], 'r-', lw=1, label="Path")  
trail_x, trail_y = [], []  


def init():
    line.set_data([], [])
    path.set_data([], [])
    trail_x.clear()  
    trail_y.clear()
    return line, path

# Animationsfunktion
def update(frame):
    x_positions = [0, x1[frame], x2[frame]]
    y_positions = [0, y1[frame], y2[frame]]

    line.set_data(x_positions, y_positions)  
    trail_x.append(x2[frame])
    trail_y.append(y2[frame])
    path.set_data(trail_x, trail_y)  

    return line, path

# Generate animation
ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=50)  # Intervall angepasst

plt.legend()
plt.title("Douple pendulum")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()
