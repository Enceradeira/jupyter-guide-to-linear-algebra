import matplotlib.pyplot as plt
import numpy as np
import subprocess
import guide_to_linear_algebra.solve_system as so

# Data points to be approximated by a polynomial
x = np.array([2, 5, 6, 15])
y = np.array([8, 12, 14, 15])

fig, ax = plt.subplots()
ax.scatter(x, y, color='red')

ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)

# Calculate the coefficients of the interpolating polynomial
A = np.zeros((4, 4))
B = np.zeros((4, 1))

for i in range(4):
    B[i, 0] = y[i]
    for j in range(4):
        A[i, j] = x[i]**(j)

print(A, '\n')
print(B)

coeffs = so.SolveSystem(A, B)
print(coeffs)

# Plot the polynomial
x_fit = np.linspace(x[0], x[3], 50)
y_fit = coeffs[0] + coeffs[1]*x_fit + coeffs[2]*x_fit**2 + coeffs[3]*x_fit**3

fig, ax = plt.subplots()

ax.scatter(x, y, color='red')
ax.plot(x_fit, y_fit, 'b')
ax.set_xlim(0, 20)
ax.set_ylim(0, 30)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)

# Open the plot
plot_path = '/home/jorg/projects/jj/jupyter-guide-to-linear-algebra/plt.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

subprocess.run(['xdg-open', plot_path])
