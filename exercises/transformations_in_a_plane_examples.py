import numpy as np
from math import pi, sin, cos
import matplotlib.pyplot as plt
import guide_to_linear_algebra.plt_helper as ph

coords = np.array([[0, 0], [0.5, 0.5], [0.5, 1.5], [0, 1], [0, 0]])
coords = coords.transpose()

x = coords[0, :]
y = coords[1, :]


def plot_coords(coords):
    x_LT1 = coords[0, :]
    y_LT1 = coords[1, :]

    fix, ax = plt.subplots()

    ax.plot(x, y, 'ro')
    ax.plot(x_LT1, y_LT1, 'bo')

    ax.plot(x, y, 'r', ls='--')
    ax.plot(x_LT1, y_LT1, 'b')

    ax.axvline(x=0, color="k", ls=":")
    ax.axhline(y=0, color="k", ls=":")
    ax.grid(True)
    ax.axis([-2, 2, -1, 2])
    ax.set_aspect('equal')


# Example 1: Stretch
A = np.array([
    [2, 0],
    [0, 1]])
A_coords = A@coords

plot_coords(A_coords)
ph.open_plt()

# Example 2: Reflection

B = np.array([
    [-1, 0],
    [0, 1]])

B_coords = B@coords
plot_coords(B_coords)
ph.open_plt()


# Example 3: Rotation
theta = pi/6
R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
R_coords = R@coords

plot_coords(R_coords)
ph.open_plt()


