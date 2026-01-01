import numpy as np
import guide_to_linear_algebra.laguide as lag
import guide_to_linear_algebra.plt_helper as ph

# Exercise 1:
# Polynomials: p1 = 3x² + 2x + 1, p2 = 2x² + 5x + 3, p3 = 6x² + 4x + 5
# Using basis {1, x, x²}, each column represents a polynomial's coefficients
P = np.array([[1, 3, 5],
              [2, 5, 4],
              [3, 2, 6]])

P_reduced = lag.FullRowReduction(P)
print("Reduced form of polynomial coefficient matrix P:", "\n", P_reduced, "\n")

# {p1, p2, p3} is a basis for the polynomial vector space since there
# is a pivot in every row.

# Exercise 2:
# Basis α = {p1, p2, p3}: p1 = x² + x + 2, p2 = 2x² + 4x, p3 = 3x² + 2x + 1
# Find coordinates of p4 = 11x² + 13x + 4 with respect to basis α
A = np.array([[2, 0, 1],
              [1, 4, 2],
              [1, 2, 3]])

b = np.array([[4], [13], [11]])
A_augmented = np.hstack((A, b))
R = lag.FullRowReduction(A_augmented)
print("Reduced form of augmented matrix [A|b]:", "\n", R, "\n")

C = np.array([[1], [2], [2]])
print("Coordinates of p4 with respect to basis α:", "\n", C, "\n")

# Exercise 4:
# Basis for a subspace of 2x2 matrices
A4 = np.array([[1, 0],
               [2, 0]])

kB4 = np.array([[4, 0],
               [5, 0]])

# Exercise 5:
# Basis β = {A, B, C, D} for M₂ₓ₂
# Find coordinates of F with respect to β
A5 = np.array([[1, 0],
               [0, 1]])

B5 = np.array([[2, 1],
               [2, 2]])

C5 = np.array([[3, 0],
               [1, 4]])

D5 = np.array([[3, 4],
               [1, 1]])

F5 = np.array([[14, 10],
               [7, 11]])

# Flatten matrices into column vectors and form augmented matrix
a5 = A5.flatten().reshape(-1, 1)
b5 = B5.flatten().reshape(-1, 1)
c5 = C5.flatten().reshape(-1, 1)
d5 = D5.flatten().reshape(-1, 1)
f5 = F5.flatten().reshape(-1, 1)

M5_augmented = np.hstack((a5, b5, c5, d5, f5))
R5 = lag.FullRowReduction(M5_augmented)
print("Reduced form of augmented matrix for Exercise 5:", "\n", R5, "\n")

Coordinates_F = R5[:, -1]
print("Coordinates of F with respect to basis β:", "\n", Coordinates_F, "\n")


# Exercise 6a:
# D2x2 is a subspace of M2x2 because any linear combination is part of M2x2

# Exercise 6b:
# Given: a Basis is linearly independent and spans the space
# Find: Find a basis for D2x2
# Solution:  let's find a basis for D2x2 in D4

B1 = np.array([[1], [0], [0], [3]])
B2 = np.array([[1], [0], [0], [1]])
Zero = np.array([[0], [0], [0], [0]])

B1B2Zero = np.hstack((B1, B2, Zero))
B1B2Zero_reduced = lag.FullRowReduction(B1B2Zero)
print("Basis vectors B1, B2, and Zero:\n", B1B2Zero, "\n")
print("Reduced form of matrix with B1, B2, and Zero:\n", B1B2Zero_reduced, "\n")

# we have two pivot columns, so B1 and B2 are linearly independent

# Transforming B1 and B2 back to matrix form
B1_matrix = B1.reshape(2, 2)
B2_matrix = B2.reshape(2, 2)

print("Basis for D2x2:")
print("B1:\n", B1_matrix)
print("B2:\n", B2_matrix)

# The dimension of D2x2 is 2 since we have two basis vectors.
