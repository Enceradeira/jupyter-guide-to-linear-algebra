import numpy as np
import guide_to_linear_algebra.solve_system as so
import matplotlib.pyplot as plt
import subprocess


def compute_interpolation_coeffs(mjx, mjy):
    """Compute interpolation coefficients for given x and y values.

    Args:
        mjx: x-coordinates
        mjy: y-coordinates

    Returns:
        coeffs: interpolation coefficients
    """
    n = len(mjx)
    A = np.zeros((n, n))
    B = np.zeros((n, 1))

    for i in range(n):
        B[i, 0] = mjy[i]
        for j in range(n):
            A[i, j] = mjx[i]**(j)

    return so.SolveSystem(A, B)


fig, ax = plt.subplots()
sequence = [
    57.9, 47.4,
    108.2, 35.0,
    149.6, 29.8,
    228.0, 24.1,
    778.5, 13.1,
    1432.0, 9.7,
    2867.0, 6.8,
    4515.0, 5.4,
]

# The data points
x = np.array(sequence[::2], dtype=float)
y = np.array(sequence[1::2], dtype=float)
ax.scatter(x, y, color='red')

# Mercury and Jupiter
mjx = np.array([x[0], x[4]])
mjy = np.array([y[0], y[4]])
coeffs = compute_interpolation_coeffs(mjx, mjy)
x_fit = np.linspace(x[0], x[7], 200)
y_fit = coeffs[0] + coeffs[1]*x_fit

ax.plot(x_fit, y_fit, 'b')
ax.set_xlim(x_fit[0]-100, x_fit[-1]+100)
ax.set_ylim(min(y_fit) - 100, max(y_fit) + 100)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)

# Mercuy, Earth, Jupiter, Saturn
mejsx = np.array([x[0], x[2], x[4], x[5]])
mejsy = np.array([y[0], y[2], y[4], y[5]])
coeffs = compute_interpolation_coeffs(mejsx, mejsy)
mesjx_fit = np.linspace(x[0], x[7], 200)
mejsy_fit = coeffs[0] + coeffs[1]*mesjx_fit + \
    coeffs[2]*mesjx_fit**2 + coeffs[3]*mesjx_fit**3
ax.plot(mesjx_fit, mejsy_fit, 'g')

# All planets
allx = x
ally = y
coeffs = compute_interpolation_coeffs(allx, ally)
allx_fit = np.linspace(x[0], x[7], 200)
ally_fit = coeffs[0] + coeffs[1]*allx_fit + \
    coeffs[2]*allx_fit**2 + coeffs[3]*allx_fit**3 + \
    coeffs[4]*allx_fit**4 + coeffs[5]*allx_fit**5 + \
    coeffs[6]*allx_fit**6 + coeffs[7]*allx_fit**7
ax.plot(allx_fit, ally_fit, 'y')


# Open the plot
plot_path = '/home/jorg/projects/jj/jupyter-guide-to-linear-algebra/plt.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

plt.close()
subprocess.run(['xdg-open', plot_path])

print("X:", mjx)
print("Y:", mjy)
