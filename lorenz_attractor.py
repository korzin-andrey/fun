import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def lorenz(x, y, z, s=5, r=28, b=2.5):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


dt = 0.01
num_steps = 1000


xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

xs[0], ys[0], zs[0] = (0., 1., 1.05)

for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')


def gen(n):
    index = 0
    phi = 0
    while index < n:
        yield np.array([xs[index], ys[index], zs[index], phi])
        index += 1
        phi += 0.1


def update(num, data, line):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])


data = np.array(list(gen(num_steps))).T
line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

# Setting the axes properties
ax.set_xlim3d([-20.0, 20.0])
ax.set_xlabel('X')

ax.set_ylim3d([-20.0, 20.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 50.0])
ax.set_zlabel('Z')
ax.view_init(np.pi*data[3][0], 60)

ani = animation.FuncAnimation(fig, update, num_steps, fargs=(
    data, line), interval=1, blit=False)
# ani.save('matplot003.gif', writer='imagemagick')
plt.show()
