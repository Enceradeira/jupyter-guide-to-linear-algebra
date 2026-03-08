import numpy as np
import guide_to_linear_algebra.laguide as lag
from guide_to_linear_algebra.inner_products import magnitude, unit_vector, angle

# Exercise 3
# Given orthonormal basis beta = {U1, U2, U3}:
# U1 = (1/sqrt(6))[2, 1, 1]^T
# U2 = (1/sqrt(2))[0, -1, 1]^T
# U3 = (1/sqrt(3))[-1, 1, 1]^T
# (a) Verify Q^T Q = I where Q has these columns
# (b) Find [X]_beta by solving Q[X]_beta = X for X = [3, 1, -4]^T
print("Exercise 3")

U1 = (1 / np.sqrt(6)) * np.array([[2], [1], [1]])
U2 = (1 / np.sqrt(2)) * np.array([[0], [-1], [1]])
U3 = (1 / np.sqrt(3)) * np.array([[-1], [1], [1]])

Q = np.hstack([U1, U2, U3])

print("Q^T Q = I:", np.allclose(Q.T @ Q, np.eye(3)))


X = np.array([[3], [1], [-4]])

b1 = lag.DotProduct(X, U1)
b2 = lag.DotProduct(X, U2)
b3 = lag.DotProduct(X, U3)
B = np.array([[b1], [b2], [b3]])

print("Q[X]_beta = X:", np.allclose(Q @ B, X))

# Exercise 4
# Find a vector orthogonal to the column space of A
print("\nExercise 4")

A = np.array([[1, 2], [2, 0], [3, 1]])
AT_aug = np.hstack([A.T, np.zeros((2, 1))])
print("Augmented system:\n", AT_aug)
R = lag.FullRowReduction(AT_aug)
print("RREF:\n", R)

x = 7
v3 = x
v2 = -1.25 * x
v1 = -0.5 * x

V = np.array([[v1], [v2], [v3]])
A1 = A[:, 0].reshape(3, 1)
A2 = A[:, 1].reshape(3, 1)
print("V·A[:,0] = 0:", lag.DotProduct(V, A1) == 0)
print("V·A[:,1] = 0:", lag.DotProduct(V, A2) == 0)

# Exercise 5
# Apply Gram-Schmidt to {V1, V2} for orthonormal basis {U1, U2}
print("\nExercise 5")

V1 = np.array([[1], [1], [2]])
V2 = np.array([[2], [1], [3]])

U1 = V1 / magnitude(V1)
V2_proj = (lag.DotProduct(V2, U1) / lag.DotProduct(U1, U1)) * U1
U2 = V2 - V2_proj
U2 = U2 / magnitude(U2)
print("[U1|U2]\n", np.hstack([U1, U2]))

print("U1·U2 = 0:", np.allclose(lag.DotProduct(U1, U2), 0))
print("||U1|| = 1:", np.allclose(magnitude(U1), 1))
print("||U2|| = 1:", np.allclose(magnitude(U2), 1))

# Exercise 6
# Apply Gram-Schmidt to {X1, X2}
print("\nExercise 6")

X1 = np.array([[1], [0], [2], [2]])
X2 = np.array([[-2], [1], [0], [-1]])

Y1 = X1 / magnitude(X1)
Y2_proj = (lag.DotProduct(X2, Y1) / lag.DotProduct(Y1, Y1)) * Y1
Y2 = X2 - Y2_proj
Y2 = Y2 / magnitude(Y2)
print("[Y1|Y2]\n", np.hstack([Y1, Y2]))

print("Y1·Y2 = 0:", np.allclose(lag.DotProduct(Y1, Y2), 0))
print("||Y1|| = 1:", np.allclose(magnitude(Y1), 1))
print("||Y2|| = 1:", np.allclose(magnitude(Y2), 1))

# Exercise 8
# Find X (projection of V onto W) and Y (orthogonal residual)
print("\nExercise 8")

V = np.array([[9], [5], [0]])
W = np.array([[3], [0], [3]])

X = (lag.DotProduct(V, W) / lag.DotProduct(W, W)) * W
Y = X - V
print("Y:\n", Y)
print("Y = X - V:", np.allclose(X - V, Y))
print("Y·W = 0:", np.allclose(lag.DotProduct(Y, W), 0))

# Exercise 9
# Generate orthogonal matrix Q from columns of A; verify Q^T Q = I
print("\nExercise 9")

A = np.array([[1, 2, -1], [0, 2, 1], [1, 1, 2]])
A1 = A[:, [0]]
Q1 = unit_vector(A1)

A2 = A[:, [1]]
Q1_hat = (lag.DotProduct(A2, Q1)/lag.DotProduct(Q1, Q1)) * Q1
Q2 = unit_vector(A2 - Q1_hat)

A3 = A[:, [2]]
Q1_hat = (lag.DotProduct(A3, Q1)/lag.DotProduct(Q1, Q1)) * Q1
Q2_hat = (lag.DotProduct(A3, Q2)/lag.DotProduct(Q2, Q2)) * Q2
Q3 = unit_vector(A3 - Q1_hat - Q2_hat)

Q = np.hstack([Q1, Q2, Q3])
print("Q:\n", Q)
print("Q^T@Q = I:", np.allclose(Q.T @ Q, np.eye(3)))

# Exercise 10
# Given Z is projection of X onto Y, and X·Y = 6, find Y
print("\nExercise 10")

X = np.array([[2], [1], [1]])
Z = np.array([[1], [1], [0]])
Y = np.array([[2], [2], [0]])

print("X·Y = 6:", np.allclose(lag.DotProduct(X, Y), 6))
