import numpy as np
import scipy.linalg as sp
import guide_to_linear_algebra.laguide as lag
from guide_to_linear_algebra.inner_products import magnitude, unit_vector
from guide_to_linear_algebra.solve_system import RowReduction

# Exercise 1
# Apply Gram-Schmidt to {V1, V2, V3} to produce an orthonormal set {U1, U2, U3}.
# (a) Without using QRFactorization or scipy.
# (b) Verify orthonormality and that they span the same space.
# (c) Compare against QRFactorization / scipy.
print("Exercise 1")

V1 = np.array([[0], [2], [0], [1]])
V2 = np.array([[1], [1], [1], [1]])
V3 = np.array([[2], [2], [0], [1]])

U1 = unit_vector(V1)
U2 = unit_vector(V2 - ((lag.DotProduct(V2, U1)/lag.DotProduct(U1, U1)) * U1))
U3 = unit_vector(V3
                 - ((lag.DotProduct(V3, U1)/lag.DotProduct(U1, U1)) * U1)
                 - ((lag.DotProduct(V3, U2)/lag.DotProduct(U2, U2)) * U2))

# a.) Verify orthonormality
print("|U1| = 1:", np.isclose(magnitude(U1), 1))
print("|U2| = 1:", np.isclose(magnitude(U2), 1))
print("|U3| = 1:", np.isclose(magnitude(U3), 1))
print("U1·U2 = 0:", np.isclose(lag.DotProduct(U1, U2), 0))
print("U1·U3 = 0:", np.isclose(lag.DotProduct(U1, U3), 0))
print("U2·U3 = 0:", np.isclose(lag.DotProduct(U2, U3), 0))

# b.) Verify that they span the same space
U_matrix = np.hstack([U1, U2, U3])
print("\nRow reduce [U1 U2 U3 | V1]:")
print(RowReduction(np.hstack([U_matrix, V1])))
print("\nRow reduce [U1 U2 U3 | V2]:")
print(RowReduction(np.hstack([U_matrix, V2])))
print("\nRow reduce [U1 U2 U3 | V3]:")
print(RowReduction(np.hstack([U_matrix, V3])))

# c.) Compare against QRFactorization and scipy
V_matrix = np.hstack([V1, V2, V3])

print("\n--- Hand-computed Q and R ---")
R_hand = U_matrix.T @ V_matrix
print("Q (U_matrix):")
print(U_matrix)
print("R (U^T @ V):")
print(np.round(R_hand, 8))

print("\n--- QRFactorization (laguide) ---")
Q_lag, R_lag = lag.QRFactorization(V_matrix)
print("Q:")
print(np.round(Q_lag, 8))
print("R:")
print(np.round(R_lag, 8))

print("\n--- scipy.linalg.qr ---")
Q_sp, R_sp = sp.qr(V_matrix, mode='economic')
print("Q:")
print(np.round(Q_sp, 8))
print("R:")
print(R_sp)

# Exercise 2
# a.) one Vector in Q will be all zeros
print("\n\nExercise 2")
B = np.array([[1, 3, -1], [0, -1, 1], [2, 2, 2], [1, 1, 1], [1, 0, 2]])

print("\n--- QRFactorization (laguide) ---")
Q_b, R_b = lag.QRFactorization(B)
print("Q:")
print(np.round(Q_b, 8))
print("R:")
print(np.round(R_b, 8))

print("\n--- scipy.linalg.qr ---")
Q_bsp, R_bsp = sp.qr(B, mode='economic')
print("Q:")
print(np.round(Q_bsp, 8))
print("R:")
print(np.round(R_bsp, 8))

# Exercise 3
# Find the QR factorization of A without using QRFactorization.
# Then verify QR = A.
print("\n\nExercise 3")
A = np.array([[1, 3, 0, 2], [0, 1, 2, 1], [2, 1, 2, 1], [1, 0, 1, 3]])

A1 = A[:, 0:1]
A2 = A[:, 1:2]
A3 = A[:, 2:3]
A4 = A[:, 3:4]

# make A1 a unit vector
W1 = unit_vector(A1)
# subtract the proejction of A2 onto W1 from A2,
# then make it a unit vector
W2 = unit_vector(A2 - ((lag.DotProduct(A2, W1)/lag.DotProduct(W1, W1)) * W1))
# subtract the projection of A3 onto W1 and W2 from A3l,
# then make it a unit vector
W3 = unit_vector(A3
                 # procection of A3 onto W1
                 - ((lag.DotProduct(A3, W1)/lag.DotProduct(W1, W1)) * W1)
                 # projection of A3 onto W2
                 - ((lag.DotProduct(A3, W2)/lag.DotProduct(W2, W2)) * W2))


# subtract the projection of A4 onto W1, W2, and W3 from A4,
W4 = unit_vector(A4
                 # projection of A4 onto W1
                 - ((lag.DotProduct(A4, W1)/lag.DotProduct(W1, W1)) * W1)
                 # projection of A4 onto W2
                 - ((lag.DotProduct(A4, W2)/lag.DotProduct(W2, W2)) * W2)
                 # projection of A4 onto W3
                 - ((lag.DotProduct(A4, W3)/lag.DotProduct(W3, W3)) * W3))

Q3 = np.hstack([W1, W2, W3, W4])
R3 = Q3.T @ A
print("Q:")
print(np.round(Q3, 8))
print("R:")
print(np.round(R3, 8))
print("QR = A:", np.allclose(Q3 @ R3, A))

# Exercise 4
# no, because in A=QR, A does not have to be square, but beeing square is a requirement for invertibility.

# Exercise 5
# Use the QR factorization of A to solve AX = B, and verify the solution.
print("\n\nExercise 5")

A5 = np.array([[1, 2, 3], [0, 3, 2], [1, 1, 4]])
B5 = np.array([[1], [1], [4]])

Q5, R5 = lag.QRFactorization(A5)
C5 = Q5.T @ B5
X5 = lag.BackSubstitution(R5, C5)
print("AX = B:", np.allclose(A5 @ X5, B5))

