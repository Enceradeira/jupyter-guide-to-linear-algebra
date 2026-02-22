import matplotlib.pyplot as plt
import numpy as np
from math import pi, sin, cos
import matplotlib
matplotlib.use('QtAgg')


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
    all_x = np.concatenate([x, x_LT1])
    all_y = np.concatenate([y, y_LT1])
    margin = 1
    x_min, x_max = int(np.floor(all_x.min())) - margin, int(np.ceil(all_x.max())) + margin
    y_min, y_max = int(np.floor(all_y.min())) - margin, int(np.ceil(all_y.max())) + margin
    ax.axis([x_min, x_max, y_min, y_max])
    ax.set_xticks(range(x_min, x_max + 1))
    ax.set_yticks(range(y_min, y_max + 1))
    ax.set_aspect('equal')


def plot_vectors_3d(trans, orig):
    n = orig.shape[1]
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    origin = [0, 0, 0]
    for i in range(n):
        c = colors[i % len(colors)]
        ax.quiver(*origin, *orig[:, i], color=c,
                  linestyle='--', label=f'e{i+1} orig')
        ax.quiver(*origin, *trans[:, i], color=c, label=f'e{i+1} trans')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-3, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-1, 2])
    ax.legend()


def plot_coords_3d(trans, orig):
    x_LT1 = trans[0, :]
    y_LT1 = trans[1, :]
    z_LT1 = trans[2, :]

    x = orig[0, :]
    y = orig[1, :]
    z = orig[2, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z, 'ro')
    ax.plot(x_LT1, y_LT1, z_LT1, 'bo')

    ax.plot(x, y, z, 'r', ls='--')
    ax.plot(x_LT1, y_LT1, z_LT1, 'b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-2, 4])
    ax.set_ylim([-2, 4])
    ax.set_zlim([-2, 4])
    # ax.view_init(elev=90, azim=-90)


# Exercise 1:
coords = np.array([
    [0, 0, 1],
    [0, 3, 1],
    [1, 3, 1],
    [1, 1, 1],
    [2, 1, 1],
    [2, 0, 1],
    [0, 0, 1]])

coords = coords.transpose()

R = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

S = np.array([
    [1, 0, -4],
    [0, 1, 0],
    [0, 0, 1]
])

trans_coords = S @ coords

# plot_coords_3d(R @ coords, coords)
# plt.show()
# plot_coords_3d(S @ coords, coords)
# plt.show()
# plot_coords_3d(trans_coords, coords)
# plt.show()
# plot_coords(trans_coords[:2, :], coords[:2, :])
# plt.show()

# Exercise 2:

T1 = np.array([
    [1, 0, -2],
    [0, 1, -1],
    [0, 0, 1]])

teta = pi / 4
R = np.array([
    [cos(teta), -sin(teta), 0],
    [sin(teta), cos(teta), 0],
    [0, 0, 1]
])

T2 = np.array([
    [1, 0, 2],
    [0, 1, 1],
    [0, 0, 1]])

# plot_coords_3d(T1 @ coords, coords)
# plot_coords_3d(T1 @ R @ coords, coords)
# plot_coords_3d(T1 @ R @ T2 @ coords, coords)
# plot_coords((T1 @ R @ T2 @ coords)[:2, :], coords[:2, :])

# std_basis_vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# plot_vectors_3d(T1 @ std_basis_vec, std_basis_vec)
# plot_vectors_3d(T1 @ R @ std_basis_vec, std_basis_vec)
# plot_vectors_3d(T1 @ R @ T2 @ std_basis_vec, std_basis_vec)

# plt.show()

# Exercise 3:

# shift by rotation center
T1 = np.array([
    [1, 0, -1],
    [0, 1, -1],
    [0, 0, 1]])

# rotate
teta = pi
R = np.array([
    [cos(teta), -sin(teta), 0],
    [sin(teta), cos(teta), 0],
    [0, 0, 1]
])

# shift back
T2 = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1]])

# translated
T3 = np.array([
    [1, 0, 3],
    [0, 1, 0],
    [0, 0, 1]])

plot_coords_3d(T1 @ coords, coords)
plot_coords_3d(R @ T1 @ coords, coords)
plot_coords_3d(T2 @ R @ T1 @ coords, coords)
plot_coords_3d(T3 @ T2 @ R @ T1 @ coords, coords)
plot_coords((T3 @ T2 @ R @ T1 @ coords)[:2, :], coords[:2, :])
plt.show()


# Exercise 4:

