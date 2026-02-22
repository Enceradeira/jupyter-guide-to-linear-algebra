from guide_to_linear_algebra.web_navigation import web_navigation_matrix
import matplotlib.pyplot as plt
import numpy as np
from math import pi, sin, cos
import matplotlib
matplotlib.use('QtAgg')


# Exercise 1
A = np.array([[0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
              [0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
              [1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
              [0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
              [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
              [1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
              [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
              [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
              [0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
              [1, 0, 1, 1, 0, 0, 1, 1, 0, 0]])

print(web_navigation_matrix(A))

# Exercise 2
A = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 0]
])

B = web_navigation_matrix(A)
print("B:", B)

X = np.array([[0], [1], [0], [0]])
steps = 100

for i in range(steps):
    X = B @ X

print("after", steps, "steps:", X)
