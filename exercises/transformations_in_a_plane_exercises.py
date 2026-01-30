import numpy as np
from math import pi, sin, cos
import matplotlib.pyplot as plt
import guide_to_linear_algebra.plt_helper as ph

coords = np.array([[0, 0], [0.5, 0.5], [0.5, 1.5], [0, 1], [0, 0]])
coords = coords.transpose()


def plot_coords(trans, orig):
    x_LT1 = trans[0, :]
    y_LT1 = trans[1, :]

    x = orig[0, :]
    y = orig[1, :]

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


# Esercise 1:
R = np.array([
    [1, 0],
    [0, -1]])

R_coords = R@coords
plot_coords(R_coords, coords)
ph.open_plt()


# Esercise 2:
S = np.array([
    [0, 1],
    [1, 0]])
S_coords = S@coords
plot_coords(S_coords, coords)
ph.open_plt()


# Esercise 3:
theta = pi/20
T = np.array([
    [cos(theta), sin(theta)],
    [-sin(theta), cos(theta)]])
T_coords = T@coords
plot_coords(T_coords, coords)
ph.open_plt()


# Esercise 4:
k = -0.25
U = np.array([
    [1, 0],
    [k, 1]])
U_coords = U@T@coords
plot_coords(U_coords, coords)
ph.open_plt()


# Esercise 5: Star of David
triangle_coords = np.array([[0, 1], [0.866, -0.5], [-0.866, -0.5], [0, 1]])
triangle_coords = triangle_coords.transpose()
theta = pi
T = np.array([
    [cos(theta), sin(theta)],
    [-sin(theta), cos(theta)]])
T_triangle_coords = T@triangle_coords
plot_coords(T_triangle_coords, triangle_coords)
ph.open_plt()

# Esercise 6
# Horizontal stretch
H = np.array([
    [2, 0],
    [0, 1]])

# Vertical stretch
V = np.array([
    [1, 0],
    [0, 2]])

U_triangle_coords = V@H@triangle_coords
plot_coords(U_triangle_coords, triangle_coords)
ph.open_plt()
