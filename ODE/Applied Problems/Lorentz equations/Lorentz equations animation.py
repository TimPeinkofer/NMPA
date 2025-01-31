import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def rk4(system, y0, t):
    n = len(t)
    h = t[1] - t[0]  # Calculate step size
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(1, n):
        k1 = system(y[i - 1], t[i - 1])
        k2 = system(y[i - 1] + h * k1 / 2, t[i - 1] + h / 2)
        k3 = system(y[i - 1] + h * k2 / 2, t[i - 1] + h / 2)
        k4 = system(y[i - 1] + h * k3, t[i - 1] + h)
        y[i] = y[i - 1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y

def lorenz_attractor(r, t):
    sigma, R, beta = 10, 28, 8 / 3
    x, y, z = r
    return np.array([
        sigma * (y - x),
        x * (R - z) - y,
        x * y - beta * z
    ])

# Time parameters
t0, t_end, h = 0, 50, 1e-3
t = np.linspace(t0, t_end, int((t_end - t0) / h) + 1)
y0 = np.array([1, 1, 1])

# Compute the solution
r = rk4(lorenz_attractor, y0, t)

# Create 3D animation
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Lorenz Attractor")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Set axis limits
ax.set_xlim(-25, 25)
ax.set_ylim(-35, 35)
ax.set_zlim(0, 50)

# Initialize line and point
line, = ax.plot([], [], [], lw=0.5, color='b')
point, = ax.plot([], [], [], 'ro')

# Add a text object for time display
time_template = 'Time = %.1f'
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    time_text.set_text('')
    return line, point, time_text

def update(frame):
    if frame >= len(r):
        raise ValueError("frame index is out of bounds")
    line.set_data(r[:frame, 0], r[:frame, 1])
    line.set_3d_properties(r[:frame, 2])
    point.set_data([r[frame, 0]], [r[frame, 1]])  # Passing sequences
    point.set_3d_properties([r[frame, 2]])        # Passing a sequence
    time_text.set_text(time_template % (frame * h))
    ax.view_init(elev=30, azim=frame * 0.1)
    return line, point, time_text

# Set the interval to 1 ms to achieve 1 second per real second
ani = FuncAnimation(fig, update, frames=len(r), init_func=init, interval=1, repeat=False)
plt.show()