import numpy as np
import guide_to_linear_algebra.laguide as lag

# Example 3
# N(A) and C(A^T) are orthogonal complements in R4

A = np.array([[2, 1, 2, 0],
              [3, 0, 1, 1],
              [1, 1, 1, 0]])
A_reduced = lag.FullRowReduction(A)
print("A_reduced:\n", A_reduced, "\n")
t = 3

X = np.array([[t], [0], [-t], [-2*t]])
print("A @ X:\n", A @ X)  # should be all zeros

# Null space basis (dimension 1)
t = -1
N_A = np.array([[t], [0], [-t], [-2*t]])
print("Null space basis:\n", N_A, "\n")
print("A @ N(A):\n", A @ N_A)  # should be all zeros

# Row space basis — the rows of A (dimension 3)
C_At = A.T

# Verify orthogonality: should be all zeros
print("Example 3")
print(C_At.T @ N_A)


## Playfield

A = np.array([[2, 1, 2],
              [3, 0, 1],
              [1, 1, 1]])
A_reduced = lag.FullRowReduction(A)
print("A_reduced:\n", A_reduced, "\n")
