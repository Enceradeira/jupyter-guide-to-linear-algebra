import numpy as np
import matplotlib.pyplot as plt
import subprocess
import guide_to_linear_algebra.solve_system as so

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([0.26, 0.20, 1.17, 2.65, 0.14, 0.42, 1.65, 2.73, 0.09, 0.21])

A = np.zeros((10, 10))
B = np.zeros((10, 1))

for i in range(10):
    B[i, 0] = y[i]
    for j in range(10):
        A[i, j] = x[i]**(j)

coeffs = so.SolveSystem(A, B)

x_fit = np.linspace(x[0], x[9], 200)
y_fit = (coeffs[0] + coeffs[1]*x_fit + coeffs[2]*x_fit**2 + coeffs[3]*x_fit**3 +
         coeffs[4]*x_fit**4 + coeffs[5]*x_fit**5 + coeffs[6]*x_fit**6 +
         coeffs[7] * x_fit**7 + coeffs[8]*x_fit**8 + coeffs[9]*x_fit**9)

fig, ax = plt.subplots()

ax.scatter(x, y, color='red')
ax.plot(x_fit, y_fit, 'b')
ax.set_xlim(0, 11)
ax.set_ylim(-7, 7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)


# Open the plot
plot_path = '/home/jorg/projects/jj/jupyter-guide-to-linear-algebra/plt.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

subprocess.run(['xdg-open', plot_path])
