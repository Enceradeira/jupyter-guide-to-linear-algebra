import matplotlib.pyplot as plt
import numpy as np
import guide_to_linear_algebra.laguide as lag
import guide_to_linear_algebra.plt_helper as ph
from guide_to_linear_algebra.general_linear_systems_functions import NrFreeVariables

# Exercise 1
A = np.array(
    [[5, 4, -1],
     [1, 0, 3],
     [-2, 2, 4],
     [1, 8, 7],
     [3, 0, -3]])
B = np.array([3, 2, -3, -3, 3])
E = np.column_stack([A, B])

E_reduced = lag.FullRowReduction(E)
print("Reduced form of matrix E:", "\n", E_reduced)
# The solutions are
# x1 = 1.25
# x2 = -0.75
# x3 = 0.25

X = np.array([[1.25], [-0.75], [0.25]])
A1 = A[:3, :]
A2 = A[2:5, :]

print("Check A1 * X (first 3 rows):", A1 @ X, '\n')
print("Check A2 * X: (last 3 rows)", A2 @ X, '\n')

# Exercise 2
A = np.array(
    [[5, 4, -1],
     [1, 0, 1],
     [-2, 2, 4],
     [1, 8, 7],
     [3, 0, -3]])
B = np.array([0, 2, -3, -3, 3])
E = np.column_stack([A, B])

E_reduced = lag.FullRowReduction(E)
print("Reduced form of matrix E:", "\n", E_reduced)

# No solutions, becaue there is a pivout in the last column

# Exercise 3
A = np.array(
    [[5, 4, -1],
     [1, 2, 1],
     [-2, 2, 4],
     [1, 8, 7],
     [3, 0, -3]])
B = np.array([3, 0, -3, -3, 3])
E = np.column_stack([A, B])

E_reduced = lag.FullRowReduction(E)
print("Reduced form of matrix E:", "\n", E_reduced)
# The solutions are
# x1 - x3 = 1
# x2 + x3 = -0.5

# Exercise 4: Construct an example of an inconsistent system with 2 equations and 4 unknowns.
E = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ])
E_reduced = lag.FullRowReduction(E)
print("Reduced form of matrix E:", "\n", E_reduced)

# Exercise 5:
A = np.array(
    [[1, 2, 3, 4],
     [5, 6, 7, 8]])
B = np.array([10, 26])
E = np.column_stack([A, B])
E_reduced = lag.FullRowReduction(E)

print("Reduced form of matrix E:", "\n", E_reduced)
# If any of the first n columns of `[A|B]` do not have a pivot, the system
# does not have a unique solution due to the existence of a free variable.
# Here n is 4, and with only 2 equations, there cannot be a pivot in each
# of the first 4 columns.

# exercise 6


A = np.array(
    [[5, 4, -1],
     [1, 0, 3],
     [-2, 2, 4],
     [1, 8, 7],
     [3, 0, -3]])
B = np.array([3, 2, -3, -3, 3])
E = np.column_stack([A, B])
print("E:", E, "\n", "Number of free variables:", NrFreeVariables(E))

# A = np.array(
#     [[5, 4, -1],
#      [1, 2, 1],
#      [-2, 2, 4],
#      [1, 8, 7],
#      [3, 0, -3]])
# B = np.array([3, 0, -3, -3, 3])
# E = np.column_stack([A, B])
# print("E:", E, "\n", "Number of free variables:", NrFreeVariables(E))

